"""
compare_models.py — Comprehensive cross-model comparison table.

Loads all available checkpoints and prints a ranked table of
MSE / overlap / PC1 on both test sets. Saves results to JSON.

Usage:
  python src/evaluation/compare_models.py
  python src/evaluation/compare_models.py --sort-by mse
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import torch

from core.data         import make_splits
from evaluation.utils  import load_registry, compute_metrics
from evaluation.config import (
    HDF_PATH, CHECKPOINT_DIR, RESULTS_DIR,
    BATCH_SIZE, build_model_registry,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf",            default=HDF_PATH)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--results-dir",    default=RESULTS_DIR)
    ap.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    ap.add_argument("--sort-by",        choices=["mse", "overlap", "pc1"],
                    default="mse")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, _ = make_splits(args.hdf, batch_size=args.batch_size)

    registry = build_model_registry(args.checkpoint_dir, args.results_dir)
    models   = load_registry(registry, device)
    print(f"\nLoaded {len(models)} models.\n")

    test_sets = {k: loaders[k] for k in ("test1", "test2") if k in loaders}

    results = {}
    for name, model in models.items():
        results[name] = {}
        for ts_name, ts_loader in test_sets.items():
            results[name][ts_name] = compute_metrics(model, ts_loader, device)

    # ── Print ranked table ────────────────────────────────────────────────────
    for ts_name in test_sets:
        reverse = (args.sort_by in ("overlap", "pc1"))
        ranked  = sorted(results.keys(),
                         key=lambda n: results[n][ts_name][args.sort_by],
                         reverse=reverse)
        print(f"\n{'='*70}")
        print(f"  {ts_name.upper()}  (sorted by {args.sort_by})")
        print(f"{'='*70}")
        print(f"  {'Rank':<5}  {'Model':<22}  {'MSE':>10}  {'Overlap':>8}  {'PC1':>6}")
        print(f"  " + "-" * 56)
        for rank, name in enumerate(ranked, 1):
            m = results[name][ts_name]
            marker = " ◀" if rank == 1 else ""
            print(f"  {rank:<5}  {name:<22}  {m['mse']:>10.6f}  "
                  f"{m['overlap']:>8.4f}  {m['pc1']:>6.4f}{marker}")

    out_path = os.path.join(args.results_dir, "model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
