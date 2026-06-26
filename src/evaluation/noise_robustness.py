"""
noise_robustness.py — Evaluate clean-trained models on whitened noisy h1_strain.

The models are trained only on clean synthetic signals. This script passes
real whitened LIGO strain through them — a genuine OOD test.

Usage:
  python src/evaluation/noise_robustness.py
  python src/evaluation/noise_robustness.py --test-set test2
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

from core.data         import make_splits, load_noisy_test_data
from evaluation.utils  import load_registry, infer_batch, mse as _mse, overlap as _overlap
from evaluation.config import (
    HDF_PATH, CHECKPOINT_DIR, PLOTS_DIR, RESULTS_DIR,
    BATCH_SIZE, build_model_registry,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf",            default=HDF_PATH)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--plots-dir",      default=PLOTS_DIR)
    ap.add_argument("--results-dir",    default=RESULTS_DIR)
    ap.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    ap.add_argument("--test-set",       choices=["test1", "test2"], default="test1")
    args = ap.parse_args()

    os.makedirs(args.plots_dir,   exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, info = make_splits(args.hdf, batch_size=args.batch_size)
    test_loader   = loaders[args.test_set]
    test_idx      = info[f"{args.test_set}_idx"]

    try:
        x_noisy = load_noisy_test_data(args.hdf, test_idx)
    except Exception as e:
        print(f"Could not load noisy strain: {e}")
        print("Is 'injection_samples/h1_strain' present in the HDF file?")
        return

    y_clean = torch.cat([y for _, y in test_loader])
    assert x_noisy.shape[0] == y_clean.shape[0], "Shape mismatch between noisy and clean"
    print(f"\nNoisy input: {x_noisy.shape}  "
          f"range=[{x_noisy.min():.2f}, {x_noisy.max():.2f}]")

    registry = build_model_registry(args.checkpoint_dir, args.results_dir)
    models   = load_registry(registry, device)
    print(f"Loaded {len(models)} models.\n")

    results = {}
    print(f"  {'Model':<22}  {'MSE':>10}  {'Overlap':>8}")
    print("  " + "-" * 44)

    for name, model in models.items():
        xhat = infer_batch(model, x_noisy, device)
        m = {
            "mse":     _mse(xhat, y_clean),
            "overlap": _overlap(xhat, y_clean),
        }
        results[name] = m
        print(f"  {name:<22}  {m['mse']:>10.6f}  {m['overlap']:>8.4f}")

    # Waveform plot for the model with best overlap
    if results:
        best_name = max(results, key=lambda k: results[k]["overlap"])
        best_model = models[best_name]
        xhat_best  = infer_batch(best_model, x_noisy, device)

        n_ex = min(4, x_noisy.shape[0])
        t    = np.arange(x_noisy.shape[-1])
        fig, axes = plt.subplots(n_ex, 3, figsize=(15, 4 * n_ex), sharex=True)
        if n_ex == 1:
            axes = axes[None, :]
        for col, title in enumerate(["Noisy Strain (OOD input)",
                                     "True Clean Signal",
                                     f"Best model: {best_name}"]):
            axes[0, col].set_title(title, fontsize=10, fontweight="bold")
        for i in range(n_ex):
            for col, d in enumerate([x_noisy[i, 0], y_clean[i, 0], xhat_best[i, 0]]):
                axes[i, col].plot(t, d.numpy(), lw=0.7)
        for ax in axes[-1]:
            ax.set_xlabel("Sample index")
        fig.suptitle(f"Noise Robustness — {args.test_set}\n"
                     "(models trained on clean signals only)",
                     fontsize=12)
        fig.tight_layout()
        path = os.path.join(args.plots_dir, f"noise_robustness_{args.test_set}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {path}")

    out_path = os.path.join(args.results_dir, f"noise_robustness_{args.test_set}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
