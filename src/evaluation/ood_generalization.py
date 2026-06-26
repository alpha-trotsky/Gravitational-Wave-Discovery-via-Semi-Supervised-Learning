"""
ood_generalization.py — Train vs test MSE comparison across all models.

Produces a bar chart and a summary table showing how well each model
generalises to both OOD test sets relative to its training MSE.

Usage:
  python src/evaluation/ood_generalization.py
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

from core.data         import make_splits
from evaluation.utils  import load_registry, collect_outputs, mse as _mse
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
    args = ap.parse_args()

    os.makedirs(args.plots_dir,   exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, _ = make_splits(args.hdf, batch_size=args.batch_size)

    registry = build_model_registry(args.checkpoint_dir, args.results_dir)
    models   = load_registry(registry, device)
    print(f"\nLoaded {len(models)} models.\n")

    results = {}
    for name, model in models.items():
        _, y_tr, xhat_tr, _ = collect_outputs(model, loaders["train_eval"], device)
        _, y_t1, xhat_t1, _ = collect_outputs(model, loaders["test1"],      device)
        _, y_t2, xhat_t2, _ = collect_outputs(model, loaders["test2"],      device)
        results[name] = {
            "train": _mse(xhat_tr, y_tr),
            "test1": _mse(xhat_t1, y_t1),
            "test2": _mse(xhat_t2, y_t2),
        }

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"  {'Model':<22}  {'Train MSE':>10}  {'Test1 MSE':>10}  "
          f"{'Test2 MSE':>10}  {'Gap1':>8}  {'Gap2':>8}")
    print("  " + "-" * 74)
    for name, m in results.items():
        gap1 = m["test1"] - m["train"]
        gap2 = m["test2"] - m["train"]
        print(f"  {name:<22}  {m['train']:>10.6f}  {m['test1']:>10.6f}  "
              f"{m['test2']:>10.6f}  {gap1:>+8.6f}  {gap2:>+8.6f}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    names  = list(results.keys())
    train_ = [results[n]["train"] for n in names]
    test1_ = [results[n]["test1"] for n in names]
    test2_ = [results[n]["test2"] for n in names]

    x     = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 6))
    ax.bar(x - width, train_, width, label="Train",  color="#4C72B0")
    ax.bar(x,         test1_, width, label="Test1",  color="#DD8452")
    ax.bar(x + width, test2_, width, label="Test2",  color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MSE")
    ax.set_title("Generalisation: Train vs OOD Test MSE\n"
                 "(test1: asymmetric intermediate-q | test2: extreme mass-ratio)")
    ax.legend()
    ax.set_ylim(0, max(max(train_), max(test1_), max(test2_)) * 1.25)
    fig.tight_layout()

    path = os.path.join(args.plots_dir, "ood_generalization.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    out_path = os.path.join(args.results_dir, "ood_generalization.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
