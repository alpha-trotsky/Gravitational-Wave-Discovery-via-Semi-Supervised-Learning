"""
phase_space.py — Latent phase-space trajectory plots.

For physics models: plots q₀ vs p₀ (canonical coordinates).
For CAE: plots latent channel 0 vs channel 1.
Generates one figure per model family, coloured by time (plasma).

Usage:
  python src/evaluation/phase_space.py
  python src/evaluation/phase_space.py --n-samples 8
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from core.data         import make_splits
from evaluation.utils  import load_registry, collect_outputs, compute_pc1_ratio
from evaluation.config import (
    HDF_PATH, CHECKPOINT_DIR, PLOTS_DIR, RESULTS_DIR,
    BATCH_SIZE, build_model_registry,
)


def _plot_trajectory_grid(name, z_all, latent_dim, is_physics,
                           n_samples, plots_dir, pc1):
    """Scatter n_samples trajectories in a grid, coloured by time."""
    n_samples = min(n_samples, z_all.shape[0])
    torch.manual_seed(1)
    idx = torch.randperm(z_all.shape[0])[:n_samples].numpy()

    cols = min(n_samples, 5)
    rows = math.ceil(n_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_list = np.array(axes).reshape(-1) if n_samples > 1 else [axes]

    xlabel = "q₀" if is_physics else "z₀"
    ylabel = "p₀" if is_physics else "z₁"

    for i, si in enumerate(idx):
        z = z_all[si].numpy()
        if is_physics:
            a, b = z[0], z[latent_dim]
        else:
            a, b = z[0], z[min(1, z.shape[0] - 1)]
        T  = a.shape[0]
        sc = axes_list[i].scatter(a, b, c=np.arange(T),
                                  cmap="plasma", s=8, rasterized=True)
        axes_list[i].set_xlabel(xlabel, fontsize=8)
        axes_list[i].set_ylabel(ylabel, fontsize=8)
        axes_list[i].set_title(f"Sample {i+1}", fontsize=9)

    for j in range(n_samples, len(axes_list)):
        axes_list[j].set_visible(False)

    fig.colorbar(sc, ax=axes_list, orientation="vertical",
                 fraction=0.02, label="Latent time index")
    fig.suptitle(f"{name}  (PC1={pc1:.3f})", fontsize=12, fontweight="bold")
    fig.tight_layout()

    safe_name = name.replace(" ", "_").replace("=", "").replace(".", "")
    path = os.path.join(plots_dir, f"phase_space_{safe_name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf",            default=HDF_PATH)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--plots-dir",      default=PLOTS_DIR)
    ap.add_argument("--results-dir",    default=RESULTS_DIR)
    ap.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    ap.add_argument("--n-samples",      type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, _ = make_splits(args.hdf, batch_size=args.batch_size)
    test_loader = loaders["test1"]

    registry = build_model_registry(args.checkpoint_dir, args.results_dir)
    models   = load_registry(registry, device)
    print(f"\nLoaded {len(models)} models.")

    for name, model in models.items():
        _, _, _, z_all = collect_outputs(model, test_loader, device)
        pc1      = compute_pc1_ratio(model, test_loader, device)
        is_phys  = getattr(model, 'physics_type', None) is not None
        ld       = model.latent_dim
        print(f"\n  {name}  PC1={pc1:.4f}")
        _plot_trajectory_grid(name, z_all, ld, is_phys,
                              args.n_samples, args.plots_dir, pc1)


if __name__ == "__main__":
    main()
