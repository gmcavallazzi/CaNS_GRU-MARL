#!/usr/bin/env python3
"""
Compare two training checkpoints to quantify how much the network changed.

Metrics computed (per-network: actor, critic, actor_target, critic_target):
  1. L2 distance:       ||θ_B - θ_A||  — total magnitude of weight change
  2. Relative L2:       ||θ_B - θ_A|| / ||θ_A||  — fractional change
  3. Cosine similarity: cos(θ_A, θ_B)  — directional alignment in weight space
  4. Per-layer breakdown: which layers moved the most (relative L2)

Usage:
    python compare_checkpoints.py checkpoint_A.pth checkpoint_B.pth
    python compare_checkpoints.py checkpoints_gru/checkpoint_ep_100.pth checkpoints_gru/checkpoint_ep_200.pth
    python compare_checkpoints.py checkpoints_gru/checkpoint_ep_100.pth checkpoints_gru/checkpoint_ep_200.pth --top-k 5
"""

import argparse
import torch
import numpy as np
from collections import OrderedDict


def flatten_params(state_dict: OrderedDict) -> torch.Tensor:
    """Flatten all parameters into a single 1D vector."""
    return torch.cat([p.flatten().float() for p in state_dict.values()])


def compare_state_dicts(sd_a: OrderedDict, sd_b: OrderedDict, name: str, top_k: int = 10):
    """Compare two state dicts and print metrics."""
    vec_a = flatten_params(sd_a)
    vec_b = flatten_params(sd_b)
    diff = vec_b - vec_a

    l2_dist = diff.norm().item()
    l2_a = vec_a.norm().item()
    rel_l2 = l2_dist / l2_a if l2_a > 0 else float('inf')
    cosine = torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()

    n_params = vec_a.numel()

    print(f"\n{'=' * 60}")
    print(f"  {name}  ({n_params:,} parameters)")
    print(f"{'=' * 60}")
    print(f"  L2 distance:        {l2_dist:.6f}")
    print(f"  Relative L2:        {rel_l2:.6f}  ({rel_l2*100:.3f}%)")
    print(f"  Cosine similarity:  {cosine:.8f}")
    print(f"  Mean abs change:    {diff.abs().mean().item():.2e}")
    print(f"  Max abs change:     {diff.abs().max().item():.2e}")

    # Per-layer breakdown
    layer_diffs = []
    for key in sd_a:
        pa = sd_a[key].flatten().float()
        pb = sd_b[key].flatten().float()
        d = (pb - pa)
        l2_layer = d.norm().item()
        norm_a = pa.norm().item()
        rel = l2_layer / norm_a if norm_a > 0 else float('inf')
        layer_diffs.append((key, l2_layer, rel, pa.numel(), d.abs().max().item()))

    layer_diffs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top-{top_k} layers by relative change:")
    print(f"  {'Layer':<45} {'Rel L2':>10} {'Abs L2':>10} {'Max Δ':>10} {'#Params':>8}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for key, l2, rel, npar, maxd in layer_diffs[:top_k]:
        print(f"  {key:<45} {rel:>10.5f} {l2:>10.6f} {maxd:>10.2e} {npar:>8,}")

    return {
        'l2': l2_dist, 'rel_l2': rel_l2, 'cosine': cosine,
        'mean_abs': diff.abs().mean().item(), 'max_abs': diff.abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two training checkpoints")
    parser.add_argument("ckpt_a", help="Path to first (earlier) checkpoint")
    parser.add_argument("ckpt_b", help="Path to second (later) checkpoint")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top-changed layers to show")
    parser.add_argument("--networks", nargs='+',
                        default=['actor', 'critic'],
                        choices=['actor', 'critic', 'actor_target', 'critic_target'],
                        help="Which networks to compare (default: actor critic)")
    args = parser.parse_args()

    ckpt_a = torch.load(args.ckpt_a, map_location='cpu', weights_only=False)
    ckpt_b = torch.load(args.ckpt_b, map_location='cpu', weights_only=False)

    print(f"Checkpoint A: {args.ckpt_a}")
    print(f"Checkpoint B: {args.ckpt_b}")

    # Unwrap if checkpoints have a 'policy' key (CaNS_DRL format)
    if 'policy' in ckpt_a:
        print(f"  Episode A: {ckpt_a.get('episode_num', '?')}, steps: {ckpt_a.get('total_steps', '?')}")
        ckpt_a = ckpt_a['policy']
    if 'policy' in ckpt_b:
        print(f"  Episode B: {ckpt_b.get('episode_num', '?')}, steps: {ckpt_b.get('total_steps', '?')}")
        ckpt_b = ckpt_b['policy']

    # Handle both raw state_dict and wrapped {'actor': ..., 'critic': ...} formats
    if 'actor' in ckpt_a:
        for net in args.networks:
            if net in ckpt_a and net in ckpt_b:
                compare_state_dicts(ckpt_a[net], ckpt_b[net], net, top_k=args.top_k)
    else:
        # Single state_dict
        compare_state_dicts(ckpt_a, ckpt_b, "model", top_k=args.top_k)


if __name__ == '__main__':
    main()
