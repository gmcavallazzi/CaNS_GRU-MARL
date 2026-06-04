"""
Train GRU-based MARL (MADDPG) for turbulent channel flow control.
Agents control wall-normal velocity (blowing/suction) in patches.
Communicates with CaNS via MPI (MPMD mode).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')

from stwEnv_gru import STWGRUParallelEnv
from models_gru import SharedPolicyMADDPG
from replay_buffer import BatchedSequenceReplayBuffer
from utils_gru import log_episode_to_tensorboard


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_noise_scale(episode, config):
    noise_cfg = config['maddpg'].get('action_noise', {})
    initial_sigma = noise_cfg.get('initial_sigma', 0.15)
    final_sigma = noise_cfg.get('final_sigma', 0.01)
    decay_episodes = noise_cfg.get('decay_episodes', 150)

    if episode < decay_episodes:
        progress = episode / max(decay_episodes, 1)
        return initial_sigma * (final_sigma / initial_sigma) ** progress
    return final_sigma


def train_gru_maddpg(config, checkpoint_dir='./checkpoints_gru',
                     logs_dir='./logs_gru', device=None, resume=False):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    gc = config['grid']
    cc = config.get('control', {})
    mc = config['maddpg']
    tc = config['training']
    rc = config.get('recurrence', {})

    patch_size = cc.get('patch_size', 4)
    ring_radius = cc.get('ring_radius', 1)
    agent_grid_nx = gc['nx'] // patch_size
    agent_grid_ny = gc['ny'] // patch_size
    n_agents = agent_grid_nx * agent_grid_ny
    obs_extent = (2 * ring_radius + 1) * patch_size
    include_prev_action = config.get('observation', {}).get('include_prev_action', True)
    n_channels = 3 if include_prev_action else 2

    print(f"=== GRU MARL Training for Channel Flow ===")
    print(f"Grid: {gc['nx']}x{gc['ny']}, patch_size={patch_size}")
    print(f"Agents: {agent_grid_nx}x{agent_grid_ny} = {n_agents}")
    print(f"Observation: ({n_channels}, {obs_extent}, {obs_extent})")
    print(f"Action: scalar per agent, scaled to "
          f"[-{cc.get('action_scaling', 0.064)}, {cc.get('action_scaling', 0.064)}]")
    print(f"Recurrence: enabled={rc.get('enabled', False)}, "
          f"hidden={rc.get('hidden_size', 64)}, "
          f"seq_len={rc.get('sequence_length', 16)}, "
          f"burn_in={rc.get('burn_in_length', 4)}")

    # Policy
    policy = SharedPolicyMADDPG(config, device=device)
    recurrent = policy.recurrent

    # Replay buffer
    obs_shape = (n_channels, obs_extent, obs_extent)
    max_buf_episodes = tc.get('buffer_max_episodes', 30)
    replay_buffer = BatchedSequenceReplayBuffer(
        max_buf_episodes, n_agents, obs_shape, act_dim_per_agent=1
    )

    # Training params
    max_episodes = tc.get('max_episodes', 500)
    episode_length = tc.get('start_episode_length', 1800)
    batch_size = mc.get('batch_size', 16)
    learning_starts = tc.get('learning_starts', 3600)
    save_freq = tc.get('save_freq', 3600)
    train_freq = mc.get('train_freq', 20)
    gradient_steps = mc.get('gradient_steps', 4)
    seq_len = rc.get('sequence_length', 16)

    # Snapshot time fracs for visualization
    snapshot_fracs = config.get('logging', {}).get('snapshot_fracs', [0.1, 0.5, 0.9])
    snapshot_steps_ep = [int(episode_length * f) for f in snapshot_fracs]

    # Resume — load checkpoint and buffer BEFORE MPI setup
    total_steps = 0
    episode_num = 0
    best_reward = float('-inf')

    if resume:
        ckpt_files = sorted(
            [f for f in os.listdir(checkpoint_dir)
             if f.startswith('checkpoint_') and f.endswith('.pth')],
            key=lambda f: int(f.split('_')[-1].split('.')[0])
        )
        if ckpt_files:
            latest = os.path.join(checkpoint_dir, ckpt_files[-1])
            sd = torch.load(latest, map_location=device, weights_only=False)
            policy.load_state_dict(sd['policy'])
            total_steps = sd.get('total_steps', 0)
            episode_num = sd.get('episode_num', 0)
            buf_path = os.path.join(checkpoint_dir, 'replay_buffer.pkl')
            if os.path.exists(buf_path):
                print(f"Loading replay buffer from {buf_path}...", flush=True)
                replay_buffer.load(buf_path)
                print(f"Replay buffer loaded: {len(replay_buffer.episodes)} episodes, "
                      f"{replay_buffer.size} transitions")
            print(f"Resumed from {latest} at step {total_steps}, ep {episode_num}")

    # Create environment (sets up MPI) — after resume loading is complete
    env = STWGRUParallelEnv(config)
    env.set_episode_length(episode_length)

    # TensorBoard
    run_name = f"gru_maddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"{logs_dir}/{run_name}")
    with open(os.path.join(logs_dir, f"{run_name}_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Init episode
    observations, infos = env.reset()
    prev_actions_arr = np.zeros((n_agents, 1), dtype=np.float32)
    hidden = policy.reset_hidden() if recurrent else None

    # Episode accumulators
    ep_reward = 0.0
    ep_steps = 0
    ep_critic_loss = 0.0
    ep_actor_loss = 0.0
    ep_loss_breakdown = {}
    ep_train_count = 0
    ep_dpdx_sum = 0.0
    prev_action_flat = np.zeros(n_agents, dtype=np.float32)
    action_history = []
    ep_snapshots = []
    hidden_steps = 0  # steps since last hidden reset

    print(f"\nStarting GRU MARL training "
          f"({max_episodes} episodes, {episode_length} steps/ep)...")

    while episode_num < max_episodes:
        sigma = compute_noise_scale(episode_num, config)

        # Reset GRU hidden every seq_len steps to match training conditions
        if recurrent and hidden_steps >= seq_len:
            hidden = policy.reset_hidden()
            hidden_steps = 0

        # Collect observations: (n_agents, C, H, W)
        obs_arr = np.stack([observations[agent] for agent in env.possible_agents])
        obs_tensor = torch.FloatTensor(obs_arr).to(device)

        # Select actions
        if total_steps < learning_starts:
            actions_flat = np.random.uniform(
                -1.0, 1.0, size=n_agents).astype(np.float32)
        else:
            if recurrent:
                actions_flat, hidden = policy.select_actions(obs_tensor, hidden)
            else:
                actions_flat, _ = policy.select_actions(obs_tensor)
            noise = np.random.normal(
                0, sigma, size=actions_flat.shape).astype(np.float32)
            actions_flat = np.clip(actions_flat + noise, -1.0, 1.0)

        # Record action history (raw scalars before env processing)
        action_history.append(actions_flat.copy())

        # Build actions dict
        actions_dict = {
            agent: np.array([actions_flat[idx]], dtype=np.float32)
            for idx, agent in enumerate(env.possible_agents)
        }

        # Step environment
        next_observations, rewards, terminations, truncations, infos = env.step(
            actions_dict)

        reward_val = np.mean([rewards[a] for a in env.possible_agents])
        done = terminations.get(env.possible_agents[0], False)
        dpdx_val = infos.get(env.possible_agents[0], {}).get('dpdx', 0.0)

        # Capture snapshots at prescribed episode fractions
        if (ep_steps + 1) in snapshot_steps_ep:
            # u_field and w_field are stored on the env after step()
            u_snap = getattr(env, 'u_obs_field', None)
            w_snap = getattr(env, 'w_obs_field', None)
            if u_snap is not None and w_snap is not None:
                ep_snapshots.append({
                    'u': env.u_obs_field.copy()
                         if hasattr(env, 'u_obs_field') else np.zeros(1),
                    'w': env.w_obs_field.copy()
                         if hasattr(env, 'w_obs_field') else np.zeros(1),
                    'action_field': env.prev_action_field.copy(),
                    'step': ep_steps + 1,
                })

        # Store in replay buffer
        next_obs_arr = np.stack([
            next_observations.get(agent, np.zeros(obs_shape, dtype=np.float32))
            for agent in env.possible_agents
        ])
        rews_arr = np.array(
            [rewards[a] for a in env.possible_agents], dtype=np.float32)
        dones_arr = np.full(n_agents, float(done), dtype=np.float32)
        acts_arr = actions_flat.reshape(n_agents, 1)

        replay_buffer.add_batch(obs_arr, acts_arr, prev_actions_arr,
                                rews_arr, next_obs_arr, dones_arr)
        prev_actions_arr = acts_arr.copy()

        # Accumulate
        ep_reward += reward_val
        ep_steps += 1
        total_steps += 1
        hidden_steps += 1
        ep_dpdx_sum += dpdx_val

        # Progress logging
        if ep_steps % 50 == 0:
            print(f"  Ep {episode_num+1} | step {ep_steps}/{episode_length} | "
                  f"dpdx={dpdx_val:.6f} | r={reward_val:+.3f} | "
                  f"sigma={sigma:.4f}", flush=True)

        # Train
        if (total_steps >= learning_starts
                and total_steps % train_freq == 0
                and len(replay_buffer.episodes) >= 1):
            for _ in range(gradient_steps):
                if recurrent and replay_buffer.size >= seq_len:
                    batch = replay_buffer.sample_sequences(
                        batch_size, seq_len, device=device)
                else:
                    batch = replay_buffer.sample(batch_size, device=device)
                c_loss, a_loss, breakdown = policy.update(batch)
                ep_critic_loss += c_loss
                ep_actor_loss += a_loss
                for k, v in breakdown.items():
                    ep_loss_breakdown[k] = ep_loss_breakdown.get(k, 0.0) + v
                ep_train_count += 1

        observations = next_observations
        prev_action_flat = actions_flat.copy()

        # ============================================================
        # Episode end
        # ============================================================
        if done:
            episode_num += 1

            avg_dpdx = ep_dpdx_sum / max(ep_steps, 1)
            dpdx_ref = config['reward'].get('dpdx_uncontrolled', config['reward']['dpdx']['uncontrolled'])
            drag_reduction = (1.0 - avg_dpdx / dpdx_ref) * 100

            # Build episode data dict for logging
            ep_data = {
                'reward': ep_reward,
                'drag_reduction_pct': drag_reduction,
                'avg_dpdx': avg_dpdx,
                'critic_loss': ep_critic_loss,
                'actor_loss': ep_actor_loss,
                'loss_breakdown': ep_loss_breakdown,
                'train_count': ep_train_count,
                'noise_sigma': sigma,
                'action_history': action_history,
                'snapshots': ep_snapshots,
                'agent_grid_nx': env.agent_grid_nx,
                'agent_grid_ny': env.agent_grid_ny,
            }
            if recurrent and hidden is not None:
                ep_data['hidden_norm'] = hidden.norm().item()

            log_episode_to_tensorboard(writer, episode_num, ep_data, config)

            print(f"Ep {episode_num:3d} | R: {ep_reward:+.3f} | "
                  f"DR: {drag_reduction:+.1f}% | avg_dpdx: {avg_dpdx:.6f} | "
                  f"sigma: {sigma:.4f} | best: {best_reward:.3f}")

            # Save best
            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save({
                    'policy': policy.state_dict(),
                    'total_steps': total_steps,
                    'episode_num': episode_num,
                }, os.path.join(checkpoint_dir, 'best_checkpoint.pth'))

            # Periodic save
            if episode_num % max(save_freq // episode_length, 1) == 0:
                torch.save({
                    'policy': policy.state_dict(),
                    'total_steps': total_steps,
                    'episode_num': episode_num,
                }, os.path.join(checkpoint_dir,
                                f'checkpoint_ep_{episode_num}.pth'))
                replay_buffer.save(
                    os.path.join(checkpoint_dir, 'replay_buffer.pkl'))

            # Reset episode
            observations, infos = env.reset()
            prev_actions_arr = np.zeros((n_agents, 1), dtype=np.float32)
            hidden = policy.reset_hidden() if recurrent else None
            hidden_steps = 0
            ep_reward = 0.0
            ep_steps = 0
            ep_critic_loss = 0.0
            ep_actor_loss = 0.0
            ep_loss_breakdown = {}
            ep_train_count = 0
            ep_dpdx_sum = 0.0
            prev_action_flat = np.zeros(n_agents, dtype=np.float32)
            action_history = []
            ep_snapshots = []

    writer.close()
    print("GRU MARL training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train GRU MARL for channel flow control')
    parser.add_argument('--config', type=str, default='config_gru.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_gru')
    parser.add_argument('--logs-dir', type=str, default='./logs_gru')
    args = parser.parse_args()

    config = load_config(args.config)
    train_gru_maddpg(config, checkpoint_dir=args.checkpoint_dir,
                     logs_dir=args.logs_dir, device=args.device,
                     resume=args.resume)
