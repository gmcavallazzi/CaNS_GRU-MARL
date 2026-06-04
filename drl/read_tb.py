"""
Read TensorBoard logs and export per-episode CSV + text summary.
Usage: python read_tb.py [--logdir logs_gru] [--csv tb_data.csv]
"""
import argparse
import os
import glob
import csv
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_latest_run(logdir):
    subdirs = sorted(glob.glob(os.path.join(logdir, "*/")))
    path = subdirs[-1] if subdirs else logdir
    ea = EventAccumulator(path)
    ea.Reload()
    return ea, path


def get_scalar_dict(ea, tag):
    """Return {step: value} dict for a scalar tag."""
    try:
        events = ea.Scalars(tag)
        return {e.step: e.value for e in events}
    except KeyError:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Read TensorBoard logs")
    parser.add_argument("--logdir", default="logs_gru")
    parser.add_argument("--csv", default="tb_data.csv", help="Output CSV path")
    args = parser.parse_args()

    ea, run_path = load_latest_run(args.logdir)
    tags = sorted(ea.Tags().get("scalars", []))

    print(f"Run: {run_path}")
    print(f"Tags: {tags}\n")

    # Define columns: (csv_header, tb_tag)
    columns = [
        ("reward", "Reward/episode"),
        ("drag_reduction_pct", "Physics/drag_reduction_pct"),
        ("avg_dpdx", "Physics/avg_dpdx"),
        ("critic_loss", "Loss/critic"),
        ("actor_total", "Loss/actor_total"),
        ("q_loss", "Loss/q_loss"),
        ("temporal", "Loss/temporal"),
        ("spatial", "Loss/spatial"),
        ("zero_mean", "Loss/zero_mean"),
        ("energy", "Loss/energy"),
        ("action_mean", "Action/mean"),
        ("action_std", "Action/std"),
        ("action_abs_max", "Action/abs_max"),
        ("temporal_change", "Action/temporal_change"),
        ("actuation_energy", "Control/actuation_energy"),
        ("noise_sigma", "Exploration/noise_sigma"),
        ("gru_hidden_norm", "GRU/hidden_norm"),
    ]

    # Load all data keyed by episode step
    data = {}
    for col_name, tb_tag in columns:
        data[col_name] = get_scalar_dict(ea, tb_tag)

    # Collect all episode numbers
    all_steps = set()
    for d in data.values():
        all_steps.update(d.keys())
    all_steps = sorted(all_steps)

    if not all_steps:
        print("No data found.")
        return

    # Write CSV
    csv_path = args.csv
    col_names = [c[0] for c in columns]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode"] + col_names)
        for ep in all_steps:
            row = [ep]
            for col_name in col_names:
                val = data[col_name].get(ep)
                row.append(f"{val:.8f}" if val is not None else "")
            writer.writerow(row)

    print(f"CSV written: {csv_path} ({len(all_steps)} episodes, {len(col_names)} columns)\n")

    # --- Text summary ---
    def section(title):
        print(f"\n{'='*60}\n  {title}\n{'='*60}")

    def stats(name, vals):
        if len(vals) == 0:
            print(f"  {name:35s}  [NO DATA]")
            return
        print(f"  {name:35s}  mean={np.mean(vals):+11.6f}  std={np.std(vals):.6f}  "
              f"min={np.min(vals):+11.6f}  max={np.max(vals):+11.6f}")

    section("FULL SUMMARY")
    for col_name in col_names:
        vals = np.array([v for v in data[col_name].values()])
        stats(col_name, vals)

    # Loss magnitude comparison
    section("LOSS BREAKDOWN (absolute magnitude, all episodes)")
    loss_cols = ["critic_loss", "actor_total", "q_loss", "temporal", "spatial", "zero_mean", "energy"]
    loss_means = {}
    for col in loss_cols:
        vals = np.array([v for v in data[col].values()])
        if len(vals) > 0:
            loss_means[col] = np.mean(np.abs(vals))

    if loss_means:
        total = sum(loss_means.values())
        for col in sorted(loss_means, key=loss_means.get, reverse=True):
            pct = 100 * loss_means[col] / total if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {col:25s}  {loss_means[col]:11.6f}  {pct:5.1f}%  {bar}")

    # Trend: first 5 vs last 5
    section("TREND: first 5 vs last 5 episodes")
    for col_name in col_names:
        vals = np.array(sorted(data[col_name].items()))
        if len(vals) >= 10:
            first5 = np.mean(vals[:5, 1])
            last5 = np.mean(vals[-5:, 1])
            change = last5 - first5
            print(f"  {col_name:35s}  first5={first5:+11.6f}  last5={last5:+11.6f}  delta={change:+11.6f}")

    print(f"\nDone. Full data in {csv_path}")


if __name__ == "__main__":
    main()
