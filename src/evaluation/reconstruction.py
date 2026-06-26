"""
reconstruction.py — Reconstruction quality: MSE, overlap, PC1 on both test sets.

Loads all available checkpoints, evaluates on test1 and test2, saves a
waveform plot for each model and a summary JSON.

Usage:
  python src/evaluation/reconstruction.py
  python src/evaluation/reconstruction.py --hdf ./output/dataset.hdf
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from core.data        import make_splits
from evaluation.utils  import load_registry, collect_outputs, compute_metrics
from evaluation.config import (
    HDF_PATH, CHECKPOINT_DIR, PLOTS_DIR, RESULTS_DIR,
    BATCH_SIZE, build_model_registry,
)


def plot_reconstructions(model, name, test_loader, device, plots_dir, n_ex=4):
    """4-sample waveform plot for one model."""
    x_all, y_all, xhat_all, _ = collect_outputs(model, test_loader, device)
    n_ex = min(n_ex, x_all.shape[0])
    t    = np.arange(x_all.shape[-1])

    fig, axes = plt.subplots(n_ex, 3, figsize=(15, 4 * n_ex), sharex=True)
    if n_ex == 1:
        axes = axes[None, :]
    for col, title in enumerate(["Input", "Target", "Reconstruction"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")
    for i in range(n_ex):
        for col, d in enumerate([x_all[i, 0], y_all[i, 0], xhat_all[i, 0]]):
            axes[i, col].plot(t, d.numpy(), lw=0.7)
    for ax in axes[-1]:
        ax.set_xlabel("Sample index")
    fig.suptitle(f"{name} — Reconstructions (test1)", fontsize=12)
    fig.tight_layout()

    safe_name = name.replace(" ", "_").replace("=", "").replace(".", "")
    path = os.path.join(plots_dir, f"recon_{safe_name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf",            default=HDF_PATH)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--plots-dir",      default=PLOTS_DIR)
    ap.add_argument("--results-dir",    default=RESULTS_DIR)
    ap.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    ap.add_argument("--no-plots",       action="store_true",
                    help="Skip waveform plots (metrics only)")
    args = ap.parse_args()

    os.makedirs(args.plots_dir,   exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, _ = make_splits(args.hdf, batch_size=args.batch_size)

    registry = build_model_registry(args.checkpoint_dir, args.results_dir)
    models   = load_registry(registry, device)
    print(f"\nLoaded {len(models)} models.\n")

    test_sets = {k: loaders[k] for k in ("test1", "test2") if k in loaders}

    all_results = {}
    header = f"  {'Model':<22}  {'Set':<6}  {'MSE':>10}  {'Overlap':>8}  {'PC1':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, model in models.items():
        all_results[name] = {}
        for ts_name, ts_loader in test_sets.items():
            m = compute_metrics(model, ts_loader, device)
            all_results[name][ts_name] = m
            print(f"  {name:<22}  {ts_name:<6}  "
                  f"{m['mse']:>10.6f}  {m['overlap']:>8.4f}  {m['pc1']:>6.4f}")

        if not args.no_plots:
            path = plot_reconstructions(
                model, name, loaders["test1"], device, args.plots_dir
            )
            print(f"    → {path}")

    out_path = os.path.join(args.results_dir, "reconstruction_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
