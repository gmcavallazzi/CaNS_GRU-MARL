#!/usr/bin/env python3
"""
Sweep all checkpoints and plot per-layer weight evolution across training.

Produces 4 figures:
  1. Per-layer relative L2 change (consecutive checkpoints) — actor heatmap
  2. Per-layer relative L2 change (consecutive checkpoints) — critic heatmap
  3. Cumulative drift from first checkpoint — actor (selected layers)
  4. Cumulative drift from first checkpoint — critic (selected layers)

Usage:
    python plot_checkpoint_evolution.py                          # defaults
    python plot_checkpoint_evolution.py --ckpt-dir checkpoints_gru --out-dir plots_ckpt
"""

import argparse
import os
import re
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ── helpers ──────────────────────────────────────────────────────────────────

def load_policy(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    ep = ckpt.get('episode_num', None)
    if 'policy' in ckpt:
        ckpt = ckpt['policy']
    return ckpt, ep


def layer_rel_l2(sd_a, sd_b):
    """Return dict  layer_name -> relative L2 change."""
    out = {}
    for key in sd_a:
        pa = sd_a[key].flatten().float()
        pb = sd_b[key].flatten().float()
        norm_a = pa.norm().item()
        if norm_a == 0:
            out[key] = 0.0
        else:
            out[key] = (pb - pa).norm().item() / norm_a
    return out


def layer_cum_l2(sd_ref, sd_cur):
    """Return dict  layer_name -> relative L2 drift from reference."""
    return layer_rel_l2(sd_ref, sd_cur)


def group_layers(layer_names):
    """Assign each layer a human-readable group for legend clarity."""
    groups = {}
    for name in layer_names:
        if 'conv' in name and 'weight' in name:
            groups[name] = name
        elif 'gru' in name:
            groups[name] = name
        elif 'fc' in name or 'feature_proj' in name or 'output' in name or 'mlp' in name:
            groups[name] = name
        elif 'norm' in name:
            groups[name] = name
        else:
            groups[name] = name
    return groups


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', default='checkpoints_gru')
    parser.add_argument('--out-dir', default='plots_ckpt')
    parser.add_argument('--networks', nargs='+', default=['actor', 'critic'])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover and sort checkpoints by episode number
    ckpt_files = []
    for f in os.listdir(args.ckpt_dir):
        m = re.match(r'checkpoint_ep_(\d+)\.pth', f)
        if m:
            ckpt_files.append((int(m.group(1)), os.path.join(args.ckpt_dir, f)))
    ckpt_files.sort(key=lambda x: x[0])

    if len(ckpt_files) < 2:
        print("Need at least 2 checkpoints.")
        return

    episodes = [ep for ep, _ in ckpt_files]
    paths = [p for _, p in ckpt_files]
    n_ckpt = len(paths)
    print(f"Found {n_ckpt} checkpoints, episodes {episodes[0]}–{episodes[-1]}")

    # Load first checkpoint as reference
    ref_policy, _ = load_policy(paths[0])

    for net_name in args.networks:
        if net_name not in ref_policy:
            print(f"  Skipping {net_name} (not in checkpoint)")
            continue

        sd_ref = ref_policy[net_name]
        # Filter to weight layers only (skip biases and norm params for heatmap clarity)
        all_layers = list(sd_ref.keys())
        weight_layers = [l for l in all_layers if 'weight' in l]
        n_layers = len(weight_layers)

        # Arrays to fill:  consecutive relative L2, cumulative drift from ep[0]
        consec_rel = np.zeros((n_ckpt - 1, n_layers))
        cumul_rel  = np.zeros((n_ckpt, n_layers))
        # Also track global metrics
        global_cosine = np.zeros(n_ckpt)
        global_rel_l2 = np.zeros(n_ckpt)

        prev_policy = ref_policy
        for i in range(n_ckpt):
            cur_policy, _ = load_policy(paths[i])
            sd_cur = cur_policy[net_name]

            # Cumulative drift from reference
            cum = layer_cum_l2(sd_ref, sd_cur)
            for j, layer in enumerate(weight_layers):
                cumul_rel[i, j] = cum[layer]

            # Global metrics vs reference
            vec_ref = torch.cat([sd_ref[k].flatten().float() for k in all_layers])
            vec_cur = torch.cat([sd_cur[k].flatten().float() for k in all_layers])
            diff = vec_cur - vec_ref
            norm_ref = vec_ref.norm().item()
            global_rel_l2[i] = diff.norm().item() / norm_ref if norm_ref > 0 else 0
            global_cosine[i] = torch.nn.functional.cosine_similarity(
                vec_ref.unsqueeze(0), vec_cur.unsqueeze(0)
            ).item()

            # Consecutive change
            if i > 0:
                sd_prev = prev_policy[net_name]
                con = layer_rel_l2(sd_prev, sd_cur)
                for j, layer in enumerate(weight_layers):
                    consec_rel[i - 1, j] = con[layer]

            prev_policy = cur_policy

            if (i + 1) % 20 == 0 or i == n_ckpt - 1:
                print(f"  {net_name}: processed {i+1}/{n_ckpt}")

        # ── Figure 1: Consecutive change heatmap ─────────────────────────
        fig, ax = plt.subplots(figsize=(14, max(5, n_layers * 0.35)))
        # Use log scale, clamp zeros
        data = np.clip(consec_rel.T, 1e-6, None)
        im = ax.imshow(data, aspect='auto', cmap='inferno',
                       norm=LogNorm(vmin=data[data > 0].min(), vmax=data.max()),
                       interpolation='nearest')
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([l.replace('.weight', '') for l in weight_layers], fontsize=8)
        # X ticks: show episode numbers
        xtick_pos = np.linspace(0, n_ckpt - 2, min(15, n_ckpt - 1), dtype=int)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels([f"{episodes[i]}→{episodes[i+1]}" for i in xtick_pos],
                           rotation=45, ha='right', fontsize=7)
        ax.set_xlabel('Episode transition')
        ax.set_ylabel('Layer')
        ax.set_title(f'{net_name} — consecutive relative L2 change per layer')
        plt.colorbar(im, ax=ax, label='Relative L2', shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'{net_name}_consecutive_heatmap.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved {net_name}_consecutive_heatmap.png")

        # ── Figure 2: Cumulative drift line plot ─────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        # Rank layers by final cumulative drift, plot top and bottom 5
        final_drift = cumul_rel[-1, :]
        ranked = np.argsort(final_drift)
        top_idx = ranked[-5:][::-1]
        bot_idx = ranked[:5]

        for j in top_idx:
            ax.plot(episodes, cumul_rel[:, j],
                    label=weight_layers[j].replace('.weight', ''), linewidth=1.5)
        for j in bot_idx:
            ax.plot(episodes, cumul_rel[:, j],
                    label=weight_layers[j].replace('.weight', ''),
                    linewidth=1.0, linestyle='--', alpha=0.7)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Relative L2 drift from first checkpoint')
        ax.set_title(f'{net_name} — cumulative weight drift (top 5 solid, bottom 5 dashed)')
        ax.legend(fontsize=7, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'{net_name}_cumulative_drift.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved {net_name}_cumulative_drift.png")

        # ── Figure 3: Global metrics (rel L2 + cosine) ──────────────────
        fig, ax1 = plt.subplots(figsize=(10, 4))
        color1 = 'tab:blue'
        ax1.plot(episodes, global_rel_l2, color=color1, linewidth=1.5)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Relative L2 drift from ep 0', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.plot(episodes, global_cosine, color=color2, linewidth=1.5, linestyle='--')
        ax2.set_ylabel('Cosine similarity to ep 0', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.set_title(f'{net_name} — global weight drift from initial checkpoint')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'{net_name}_global_drift.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved {net_name}_global_drift.png")

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
