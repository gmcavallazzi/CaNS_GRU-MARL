#!/usr/bin/env python3
"""
Evaluate a checkpoint by running one episode and producing diagnostic plots.

Must be launched via MPMD like training:
    mpirun --bind-to none --mca coll ^hcoll \
      -n 1 python eval_checkpoint.py --checkpoint checkpoints_gru/best_checkpoint.pth : \
      -n 1 ./cans

Produces (in --out-dir):
  1. dpdx_timeseries.png   — dp/dx vs timestep with running mean and uncontrolled baseline
  2. snapshot_fields.png   — u, w, action fields at snapshot_fracs of the episode
  3. action_heatmap.png    — action field at start/mid/end + time-averaged
  4. agent_action_trace.png — time series of a random agent + neighbors
  5. drag_reduction.png    — instantaneous drag reduction % vs timestep
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stwEnv_gru import STWGRUParallelEnv
from models_gru import SharedPolicyMADDPG
from utils_gru import plot_snapshot_grid, plot_action_heatmap, plot_agent_action_history


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a GRU-MARL checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pth')
    parser.add_argument('--config', default='config_gru.yaml', help='Config YAML')
    parser.add_argument('--out-dir', default='eval_plots', help='Output directory')
    parser.add_argument('--episode-length', type=int, default=None,
                        help='Override episode length (default: from config)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--snapshot-fracs', nargs='+', type=float, default=None,
                        help='Episode fractions for field snapshots (default: from config)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    episode_length = args.episode_length or config['training']['start_episode_length']
    snapshot_fracs = args.snapshot_fracs or config.get('logging', {}).get('snapshot_fracs', [0.1, 0.5, 0.9])
    snapshot_steps = set(int(episode_length * f) for f in snapshot_fracs)

    # Load policy
    policy = SharedPolicyMADDPG(config, device=device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt['policy'])
    ep_num = ckpt.get('episode_num', '?')
    total_steps = ckpt.get('total_steps', '?')
    print(f"Loaded checkpoint: {args.checkpoint} (episode {ep_num}, steps {total_steps})")

    recurrent = policy.recurrent
    n_agents = policy.n_agents

    # Create environment (sets up MPI with CaNS)
    env = STWGRUParallelEnv(config)
    env.set_episode_length(episode_length)

    rc = config.get('recurrence', {})
    seq_len = rc.get('sequence_length', 16)

    # ── Run one episode (no noise, no training) ──────────────────────────
    observations, infos = env.reset()
    hidden = policy.reset_hidden() if recurrent else None
    hidden_steps = 0

    dpdx_history = []
    reward_history = []
    action_history = []
    snapshots = []

    print(f"Running evaluation episode ({episode_length} steps)...")

    for step in range(episode_length):
        # Reset GRU hidden every seq_len steps (match training)
        if recurrent and hidden_steps >= seq_len:
            hidden = policy.reset_hidden()
            hidden_steps = 0

        obs_arr = np.stack([observations[a] for a in env.possible_agents])
        obs_tensor = torch.FloatTensor(obs_arr).to(device)

        # Deterministic action (no noise)
        with torch.no_grad():
            if recurrent:
                actions_flat, hidden = policy.select_actions(obs_tensor, hidden)
            else:
                actions_flat, _ = policy.select_actions(obs_tensor)

        action_history.append(actions_flat.copy())

        actions_dict = {
            agent: np.array([actions_flat[idx]], dtype=np.float32)
            for idx, agent in enumerate(env.possible_agents)
        }

        observations, rewards, terminations, truncations, infos = env.step(actions_dict)

        dpdx_val = infos[env.possible_agents[0]].get('dpdx', 0.0)
        reward_val = np.mean([rewards[a] for a in env.possible_agents])
        dpdx_history.append(dpdx_val)
        reward_history.append(reward_val)
        hidden_steps += 1

        # Snapshots
        if (step + 1) in snapshot_steps:
            u_snap = getattr(env, 'u_obs_field', None)
            w_snap = getattr(env, 'w_obs_field', None)
            if u_snap is not None and w_snap is not None:
                snapshots.append({
                    'u': u_snap.copy(),
                    'w': w_snap.copy(),
                    'action_field': env.prev_action_field.copy(),
                    'step': step + 1,
                })

        if (step + 1) % 100 == 0:
            print(f"  step {step+1}/{episode_length} | "
                  f"dpdx={dpdx_val:.6f} | r={reward_val:+.3f}")

    dpdx_arr = np.array(dpdx_history)
    reward_arr = np.array(reward_history)
    steps = np.arange(1, episode_length + 1)

    dpdx_uncontrolled = config['reward']['dpdx_uncontrolled']
    avg_dpdx = dpdx_arr.mean()
    drag_reduction = (1.0 - avg_dpdx / dpdx_uncontrolled) * 100

    print(f"\n{'='*50}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episode length: {episode_length}")
    print(f"  Avg dp/dx:      {avg_dpdx:.6f}")
    print(f"  Uncontrolled:   {dpdx_uncontrolled:.6f}")
    print(f"  Drag reduction: {drag_reduction:+.1f}%")
    print(f"  Total reward:   {reward_arr.sum():+.1f}")
    print(f"{'='*50}")

    ckpt_label = os.path.basename(args.checkpoint).replace('.pth', '')

    # ── Plot 1: dp/dx time series ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, dpdx_arr, alpha=0.4, linewidth=0.5, color='tab:blue', label='instantaneous')

    # Running mean
    window = min(50, episode_length // 10)
    if window > 1:
        kernel = np.ones(window) / window
        running_mean = np.convolve(dpdx_arr, kernel, mode='valid')
        rm_steps = steps[window - 1:]
        ax.plot(rm_steps, running_mean, color='tab:blue', linewidth=1.5,
                label=f'running mean (w={window})')

    ax.axhline(y=dpdx_uncontrolled, color='tab:red', linestyle='--', linewidth=1.5,
               label=f'uncontrolled ({dpdx_uncontrolled:.4f})')
    ax.axhline(y=avg_dpdx, color='tab:green', linestyle='-', linewidth=1.5,
               label=f'episode mean ({avg_dpdx:.5f})')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('dp/dx')
    ax.set_title(f'dp/dx evolution — {ckpt_label} (DR: {drag_reduction:+.1f}%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'dpdx_timeseries.png'), dpi=150)
    plt.close(fig)
    print("Saved dpdx_timeseries.png")

    # ── Plot 2: Instantaneous drag reduction % ───────────────────────────
    dr_inst = (1.0 - dpdx_arr / dpdx_uncontrolled) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, dr_inst, alpha=0.4, linewidth=0.5, color='tab:green', label='instantaneous')

    if window > 1:
        dr_running = np.convolve(dr_inst, kernel, mode='valid')
        ax.plot(rm_steps, dr_running, color='tab:green', linewidth=1.5,
                label=f'running mean (w={window})')

    ax.axhline(y=0, color='tab:red', linestyle='--', linewidth=1,
               label='uncontrolled baseline')
    ax.axhline(y=drag_reduction, color='black', linestyle='-', linewidth=1.5,
               label=f'episode mean ({drag_reduction:+.1f}%)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Drag reduction (%)')
    ax.set_title(f'Drag reduction — {ckpt_label}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'drag_reduction.png'), dpi=150)
    plt.close(fig)
    print("Saved drag_reduction.png")

    # ── Plot 3: Snapshot fields (u, w, action) ───────────────────────────
    if snapshots:
        img = plot_snapshot_grid(snapshots, episode=ep_num)
        if img is not None:
            fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
            ax.imshow(img)
            ax.axis('off')
            fig.tight_layout(pad=0)
            fig.savefig(os.path.join(args.out_dir, 'snapshot_fields.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
            print("Saved snapshot_fields.png")

    # ── Plot 4: Action heatmap ───────────────────────────────────────────
    if action_history:
        img = plot_action_heatmap(action_history, env.agent_grid_nx,
                                  env.agent_grid_ny, episode=ep_num)
        if img is not None:
            fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
            ax.imshow(img)
            ax.axis('off')
            fig.tight_layout(pad=0)
            fig.savefig(os.path.join(args.out_dir, 'action_heatmap.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
            print("Saved action_heatmap.png")

    # ── Plot 5: Agent action traces ──────────────────────────────────────
    if action_history:
        img = plot_agent_action_history(action_history, env.agent_grid_nx,
                                        env.agent_grid_ny, episode=ep_num)
        if img is not None:
            fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
            ax.imshow(img)
            ax.axis('off')
            fig.tight_layout(pad=0)
            fig.savefig(os.path.join(args.out_dir, 'agent_action_trace.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
            print("Saved agent_action_trace.png")

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
