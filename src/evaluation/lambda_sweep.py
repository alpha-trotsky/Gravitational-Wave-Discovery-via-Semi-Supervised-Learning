"""
lambda_sweep.py — Plot reconstruction MSE, overlap, and PC1 vs lambda.

Loads pre-trained checkpoints for HAE, ProperHAE, and PortHAE at all lambda
values and generates a 3-panel figure per model family. CAE is shown as a
horizontal reference line.

Usage:
  python src/evaluation/lambda_sweep.py
  python src/evaluation/lambda_sweep.py --families hae properhae
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from core.data         import make_splits
from evaluation.utils  import load_model, collect_outputs, mse as _mse, overlap as _overlap, compute_pc1_ratio
from evaluation.config import (
    HDF_PATH, CHECKPOINT_DIR, PLOTS_DIR, RESULTS_DIR,
    BATCH_SIZE, LAMBDA_VALUES, ckpt_path, build_model_registry,
)


FAMILY_SPECS = {
    "hae":       ("HAE",       "hae"),
    "properhae": ("ProperHAE", "properhae"),
    "porthae":   ("PortHAE",   "porthae"),
}


def _sweep_family(family_key, display_name, loaders, device,
                  checkpoint_dir, results_dir, lambda_values):
    """Collect metrics for one model family across all lambdas."""
    from core.models import HAE, ProperHAE, PortHAE
    cls_map = {"hae": HAE, "properhae": ProperHAE, "porthae": PortHAE}
    cls = cls_map[family_key]

    cfg_path = os.path.join(results_dir, "best_configs.json")
    latent_dim = 2
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            latent_dim = int(json.load(f).get(family_key, {}).get("latent_dim", 2))

    test_loader = loaders["test1"]
    rows = []
    for lam in lambda_values:
        path = ckpt_path(family_key, lam, checkpoint_dir)
        model = load_model(cls, latent_dim, path, device)
        if model is None:
            print(f"  [skip] {display_name} λ={lam} — checkpoint not found")
            continue
        _, y, xhat, _ = collect_outputs(model, test_loader, device)
        rows.append({
            "lambda":  lam,
            "mse":     _mse(xhat, y),
            "overlap": _overlap(xhat, y),
            "pc1":     compute_pc1_ratio(model, test_loader, device),
        })
        print(f"  {display_name} λ={lam}: mse={rows[-1]['mse']:.6f}  "
              f"overlap={rows[-1]['overlap']:.4f}  pc1={rows[-1]['pc1']:.4f}")
    return rows


def _plot_sweep(family_key, display_name, rows, cae_metrics, plots_dir):
    if not rows:
        print(f"  No data for {display_name} — skipping plot.")
        return

    lambdas  = [r["lambda"]  for r in rows]
    mses     = [r["mse"]     for r in rows]
    overlaps = [r["overlap"] for r in rows]
    pc1s     = [r["pc1"]     for r in rows]

    cae_mse     = cae_metrics.get("mse")
    cae_overlap = cae_metrics.get("overlap")
    cae_pc1     = cae_metrics.get("pc1")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_kw = dict(marker="o", markersize=8, linewidth=2, color="#2171B5", label=display_name)
    ref_kw   = dict(color="#D94801", linewidth=1.5, linestyle="--", label="CAE (reference)")

    panels = [
        (axes[0], mses,     cae_mse,     "Test MSE",         "MSE"),
        (axes[1], overlaps, cae_overlap, "Waveform Overlap",  "Overlap"),
        (axes[2], pc1s,     cae_pc1,     "Latent PC1 Ratio", "PC1 ratio"),
    ]
    for ax, vals, cae_val, title, ylabel in panels:
        ax.semilogx(lambdas, vals, **model_kw)
        if cae_val is not None:
            ax.axhline(cae_val, **ref_kw)
        ax.set_xlabel("λ (physics loss weight)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(lambdas)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle(f"{display_name} — Effect of λ on Test Performance",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, f"lambda_sweep_{family_key}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf",            default=HDF_PATH)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--plots-dir",      default=PLOTS_DIR)
    ap.add_argument("--results-dir",    default=RESULTS_DIR)
    ap.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    ap.add_argument("--families",       nargs="+",
                    choices=list(FAMILY_SPECS.keys()),
                    default=list(FAMILY_SPECS.keys()))
    args = ap.parse_args()

    os.makedirs(args.plots_dir,   exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, _ = make_splits(args.hdf, batch_size=args.batch_size)

    # Load CAE reference metrics
    from core.models import BaselineCAE
    cae_path = ckpt_path("cae", checkpoint_dir=args.checkpoint_dir)
    cae_ld   = 2
    cfg_file = os.path.join(args.results_dir, "best_configs.json")
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cae_ld = int(json.load(f).get("cae", {}).get("latent_dim", 2))
    cae = load_model(BaselineCAE, cae_ld, cae_path, device)
    cae_metrics = {}
    if cae is not None:
        _, y, xhat, _ = collect_outputs(cae, loaders["test1"], device)
        cae_metrics = {
            "mse":     _mse(xhat, y),
            "overlap": _overlap(xhat, y),
            "pc1":     compute_pc1_ratio(cae, loaders["test1"], device),
        }
        print(f"CAE reference: mse={cae_metrics['mse']:.6f}  "
              f"overlap={cae_metrics['overlap']:.4f}  pc1={cae_metrics['pc1']:.4f}\n")
    else:
        print("CAE checkpoint not found — will plot without reference line.\n")

    all_results = {}
    for key in args.families:
        display_name, family_key = FAMILY_SPECS[key]
        print(f"\n── {display_name} ──")
        rows = _sweep_family(family_key, display_name, loaders, device,
                             args.checkpoint_dir, args.results_dir, LAMBDA_VALUES)
        _plot_sweep(family_key, display_name, rows, cae_metrics, args.plots_dir)
        all_results[family_key] = rows

    out_path = os.path.join(args.results_dir, "lambda_sweep.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
