#!/usr/bin/env python3
"""
Analyse whether weight updates are directionally consistent (converging toward
a target) or oscillating / wandering around a basin.

Produces per-network (actor, critic):
  1. Displacement-vs-path-length ratio per layer over training.
     Ratio = 1 means straight-line march; ratio → 0 means random walk / oscillation.
  2. Consecutive update cosine similarity per layer (are successive steps aligned?).
  3. Summary panel: global displacement ratio + per-layer final ratios as bar chart.

Usage:
    python plot_checkpoint_direction.py
    python plot_checkpoint_direction.py --ckpt-dir checkpoints_gru --out-dir plots_ckpt
"""

import argparse
import os
import re

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_policy(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'policy' in ckpt:
        ckpt = ckpt['policy']
    return ckpt


def get_layer_vec(sd, layer):
    return sd[layer].flatten().float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', default='checkpoints_gru')
    parser.add_argument('--out-dir', default='plots_ckpt')
    parser.add_argument('--networks', nargs='+', default=['actor', 'critic'])
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Discover checkpoints
    ckpt_files = []
    for f in os.listdir(args.ckpt_dir):
        m = re.match(r'checkpoint_ep_(\d+)\.pth', f)
        if m:
            ckpt_files.append((int(m.group(1)), os.path.join(args.ckpt_dir, f)))
    ckpt_files.sort()
    episodes = [ep for ep, _ in ckpt_files]
    paths = [p for _, p in ckpt_files]
    n = len(paths)
    print(f"Found {n} checkpoints, episodes {episodes[0]}–{episodes[-1]}")

    if n < 3:
        print("Need at least 3 checkpoints.")
        return

    for net_name in args.networks:
        print(f"\nProcessing {net_name}...")
        sd0 = load_policy(paths[0])[net_name]
        weight_layers = [l for l in sd0 if 'weight' in l]
        n_layers = len(weight_layers)

        # ── Accumulate per-layer vectors ─────────────────────────────────
        # path_length[i, j] = sum of consecutive step norms up to checkpoint i
        # displacement[i, j] = ||θ_i - θ_0|| for layer j
        # update_cosine[i, j] = cos(Δ_{i-1→i}, Δ_{i→i+1})
        path_length = np.zeros((n, n_layers))
        displacement = np.zeros((n, n_layers))
        update_cosine = np.full((n, n_layers), np.nan)  # defined for i in [1, n-2]

        # Also track full-network (all params concatenated)
        path_length_global = np.zeros(n)
        displacement_global = np.zeros(n)
        update_cosine_global = np.full(n, np.nan)

        prev_sd = sd0
        prev_deltas = None  # per-layer deltas from step i-1 → i

        for i in range(1, n):
            cur_sd = load_policy(paths[i])[net_name]

            cur_deltas = {}
            global_delta = []
            global_prev_delta = []

            for j, layer in enumerate(weight_layers):
                v0 = get_layer_vec(sd0, layer)
                vp = get_layer_vec(prev_sd, layer)
                vc = get_layer_vec(cur_sd, layer)

                delta = vc - vp
                cur_deltas[layer] = delta

                step_norm = delta.norm().item()
                path_length[i, j] = path_length[i - 1, j] + step_norm
                displacement[i, j] = (vc - v0).norm().item()

                # Cosine between consecutive updates
                if prev_deltas is not None:
                    pd = prev_deltas[layer]
                    pd_norm = pd.norm().item()
                    d_norm = delta.norm().item()
                    if pd_norm > 0 and d_norm > 0:
                        update_cosine[i - 1, j] = torch.dot(pd, delta).item() / (pd_norm * d_norm)

                global_delta.append(delta)
                if prev_deltas is not None:
                    global_prev_delta.append(prev_deltas[layer])

            # Global
            gd = torch.cat(global_delta)
            step_g = gd.norm().item()
            path_length_global[i] = path_length_global[i - 1] + step_g

            v0_all = torch.cat([get_layer_vec(sd0, l) for l in weight_layers])
            vc_all = torch.cat([get_layer_vec(cur_sd, l) for l in weight_layers])
            displacement_global[i] = (vc_all - v0_all).norm().item()

            if prev_deltas is not None:
                gpd = torch.cat(global_prev_delta)
                gpd_n = gpd.norm().item()
                gd_n = gd.norm().item()
                if gpd_n > 0 and gd_n > 0:
                    update_cosine_global[i - 1] = torch.dot(gpd, gd).item() / (gpd_n * gd_n)

            prev_deltas = cur_deltas
            prev_sd = cur_sd

            if (i + 1) % 20 == 0 or i == n - 1:
                print(f"  {i+1}/{n}")

        # Displacement / path-length ratio (efficiency of the walk)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(path_length > 0, displacement / path_length, 1.0)
            ratio_global = np.where(path_length_global > 0,
                                    displacement_global / path_length_global, 1.0)

        short_names = [l.replace('.weight', '') for l in weight_layers]

        # ── Figure 1: Displacement ratio over training ───────────────────
        # Rank by final ratio to pick interesting layers
        final_ratio = ratio[-1, :]
        ranked = np.argsort(final_ratio)
        most_directed = ranked[-5:][::-1]   # highest ratio = most directional
        most_wandering = ranked[:5]          # lowest ratio = most oscillatory

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax = axes[0]
        ax.plot(episodes, ratio_global, 'k-', linewidth=2, label='all weights')
        for j in most_directed:
            ax.plot(episodes, ratio[:, j], linewidth=1.2, label=short_names[j])
        ax.set_ylabel('Displacement / Path length')
        ax.set_title(f'{net_name} — most directional layers (ratio → 1 = straight line)')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5)

        ax = axes[1]
        ax.plot(episodes, ratio_global, 'k-', linewidth=2, label='all weights')
        for j in most_wandering:
            ax.plot(episodes, ratio[:, j], linewidth=1.2, label=short_names[j])
        ax.set_ylabel('Displacement / Path length')
        ax.set_xlabel('Episode')
        ax.set_title(f'{net_name} — most wandering layers (ratio → 0 = oscillating)')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'{net_name}_displacement_ratio.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved {net_name}_displacement_ratio.png")

        # ── Figure 2: Update cosine similarity ───────────────────────────
        # Smooth with a rolling mean for readability
        window = 5

        def rolling_mean(arr, w):
            out = np.full_like(arr, np.nan)
            for i in range(len(arr)):
                lo = max(0, i - w // 2)
                hi = min(len(arr), i + w // 2 + 1)
                vals = arr[lo:hi]
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0:
                    out[i] = valid.mean()
            return out

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax = axes[0]
        ax.plot(episodes, rolling_mean(update_cosine_global, window),
                'k-', linewidth=2, label='all weights')
        for j in most_directed:
            ax.plot(episodes, rolling_mean(update_cosine[:, j], window),
                    linewidth=1.0, alpha=0.8, label=short_names[j])
        ax.axhline(y=0, color='grey', linestyle=':', alpha=0.5)
        ax.set_ylabel('Cosine(Δᵢ, Δᵢ₊₁)')
        ax.set_title(f'{net_name} — consecutive update direction similarity '
                     f'(rolling mean, w={window})\n'
                     f'positive = consistent direction, negative = reversing')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(episodes, rolling_mean(update_cosine_global, window),
                'k-', linewidth=2, label='all weights')
        for j in most_wandering:
            ax.plot(episodes, rolling_mean(update_cosine[:, j], window),
                    linewidth=1.0, alpha=0.8, label=short_names[j])
        ax.axhline(y=0, color='grey', linestyle=':', alpha=0.5)
        ax.set_ylabel('Cosine(Δᵢ, Δᵢ₊₁)')
        ax.set_xlabel('Episode')
        ax.set_title(f'{net_name} — consecutive update direction (wandering layers)')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'{net_name}_update_cosine.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved {net_name}_update_cosine.png")

        # ── Figure 3: Summary bar chart — final displacement ratio ───────
        fig, ax = plt.subplots(figsize=(10, max(4, n_layers * 0.3)))
        order = np.argsort(final_ratio)
        colors = plt.cm.RdYlGn(final_ratio[order])  # red=low(wandering), green=high(directed)
        ax.barh(range(n_layers), final_ratio[order], color=colors)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([short_names[i] for i in order], fontsize=8)
        ax.set_xlabel('Final displacement / path length')
        ax.set_title(f'{net_name} — per-layer directional efficiency\n'
                     f'(1 = straight march, 0 = wandering/oscillating)')
        ax.axvline(x=0.5, color='grey', linestyle=':', alpha=0.5)
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'{net_name}_direction_summary.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved {net_name}_direction_summary.png")

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
