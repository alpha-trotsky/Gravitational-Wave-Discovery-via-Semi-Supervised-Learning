"""
evaluate_ood.py — Evaluate all saved checkpoints on the two new OOD test sets.

Test sets (defined in data.py make_splits):
  test1 : mass2 in [5,15]  & mass1 in [50,65]  (asymmetric intermediate mass-ratio)
  test2 : mass2 <= 20      & mass1 > 65         (extreme mass-ratio, heavy primary)

For each test set and each available checkpoint, reports:
  - Reconstruction MSE
  - Waveform overlap (cosine similarity)
  - Latent PC1 ratio
  - Noise robustness MSE / overlap  (if h1_strain is present in the HDF)

Results written to {output_dir}/ood_metrics.json and printed as a summary table.

Usage:
  python src/evaluate_ood.py
  python src/evaluate_ood.py --hdf ./output/dataset.hdf --checkpoint-dir ./checkpoints
  python src/evaluate_ood.py --no-noise --output-dir ./results
"""

import argparse
import json
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data    import make_splits, load_noisy_test_data
from models  import HAE, BaselineCAE, ProperHAE, PortHAE
from evaluate import _collect_outputs, _infer_batch, _mse, _overlap, compute_pc1_ratio


DEFAULT_HDF      = "./output/dataset.hdf"
DEFAULT_CKPT_DIR = "./checkpoints"
DEFAULT_OUT_DIR  = "./results"


def _load_model(cls, latent_dim, ckpt_path, device):
    """Instantiate and load weights. Returns None if checkpoint is missing."""
    if not os.path.exists(ckpt_path):
        return None
    model = cls(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def _eval_model(model, loader, test_idx, hdf_path, device, with_noise):
    """Compute MSE, overlap, PC1, and optionally noise robustness for one model/loader pair."""
    _, y, xhat, _ = _collect_outputs(model, loader, device)
    result = {
        "mse":     _mse(xhat, y),
        "overlap": _overlap(xhat, y),
        "pc1":     compute_pc1_ratio(model, loader, device),
    }

    if with_noise:
        try:
            x_noisy = load_noisy_test_data(hdf_path, test_idx)
            y_clean = torch.cat([yb for _, yb in loader])
            xhat_noisy = _infer_batch(model, x_noisy, device)
            result["noise_mse"]     = _mse(xhat_noisy, y_clean)
            result["noise_overlap"] = _overlap(xhat_noisy, y_clean)
        except Exception as e:
            result["noise_mse"]     = None
            result["noise_overlap"] = None
            print(f"    [noise eval skipped: {e}]")

    return result


def main():
    ap = argparse.ArgumentParser(description="OOD evaluation on test1 and test2 splits")
    ap.add_argument("--hdf",            default=DEFAULT_HDF,
                    help="Path to dataset HDF5 file")
    ap.add_argument("--checkpoint-dir", default=DEFAULT_CKPT_DIR,
                    help="Directory containing .pt checkpoint files")
    ap.add_argument("--output-dir",     default=DEFAULT_OUT_DIR,
                    help="Directory to write ood_metrics.json")
    ap.add_argument("--batch-size",     type=int, default=16)
    ap.add_argument("--no-noise",       action="store_true",
                    help="Skip noise robustness evaluation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, info = make_splits(args.hdf, batch_size=args.batch_size)

    # Read best latent_dim values from hparam search if available
    hae_ld = cae_ld = 2
    cfg_path = os.path.join(args.output_dir, "best_configs.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        hae_ld = cfg.get("hae", {}).get("latent_dim", 2)
        cae_ld = cfg.get("cae", {}).get("latent_dim", 2)
        print(f"Loaded best configs: HAE latent_dim={hae_ld}, CAE latent_dim={cae_ld}")
    else:
        print(f"best_configs.json not found at {cfg_path}, using latent_dim=2 for all models")

    ckpt = args.checkpoint_dir

    # (display_name, model_class, latent_dim, checkpoint_filename)
    model_registry = [
        ("HAE (best)",      HAE,         hae_ld, "hae_best.pt"),
        ("CAE (best)",      BaselineCAE, cae_ld, "cae_best.pt"),
        ("HAE λ=0.01",      HAE,         2,      "hae_lambda0.01_best.pt"),
        ("HAE λ=0.1",       HAE,         2,      "hae_lambda0.1_best.pt"),
        ("HAE λ=1.0",       HAE,         2,      "hae_lambda1.0_best.pt"),
        ("ProperHAE λ=0.1", ProperHAE,   2,      "properhae_lambda0.1_best.pt"),
        ("ProperHAE λ=0.5", ProperHAE,   2,      "properhae_lambda0.5_best.pt"),
        ("ProperHAE λ=1.0", ProperHAE,   2,      "properhae_lambda1.0_best.pt"),
        ("PortHAE λ=0.1",   PortHAE,     2,      "porthae_lambda0.1_best.pt"),
        ("PortHAE λ=0.5",   PortHAE,     2,      "porthae_lambda0.5_best.pt"),
        ("PortHAE λ=1.0",   PortHAE,     2,      "porthae_lambda1.0_best.pt"),
    ]

    test_sets = [
        ("test1", loaders["test1"], info["test1_idx"],
         "mass2 in [5,15] & mass1 in [50,65]  (asymmetric intermediate-q)"),
        ("test2", loaders["test2"], info["test2_idx"],
         "mass2 <= 20 & mass1 > 65  (extreme mass-ratio, heavy primary)"),
    ]

    with_noise = not args.no_noise
    all_results = {}

    for set_name, loader, test_idx, description in test_sets:
        n = len(test_idx)
        print(f"\n{'='*65}")
        print(f"  {set_name.upper()} — {description}")
        print(f"  N = {n} samples")
        print(f"{'='*65}")

        if n == 0:
            print("  No samples in this split — skipping.")
            all_results[set_name] = {"description": description, "n_samples": 0, "models": {}}
            continue

        set_results = {}
        for model_name, cls, latent_dim, ckpt_name in model_registry:
            ckpt_path = os.path.join(ckpt, ckpt_name)
            model = _load_model(cls, latent_dim, ckpt_path, device)
            if model is None:
                continue

            metrics = _eval_model(model, loader, test_idx, args.hdf, device, with_noise)
            set_results[model_name] = metrics

            noise_str = ""
            if with_noise and metrics.get("noise_mse") is not None:
                noise_str = (f"  noise_mse={metrics['noise_mse']:.6f}"
                             f"  noise_ov={metrics['noise_overlap']:.4f}")
            print(f"  {model_name:20s}  mse={metrics['mse']:.6f}"
                  f"  overlap={metrics['overlap']:.4f}"
                  f"  pc1={metrics['pc1']:.4f}{noise_str}")

        # Print best model for this split
        if set_results:
            best = min(set_results, key=lambda k: set_results[k]["mse"])
            print(f"\n  Best by MSE: {best}  (mse={set_results[best]['mse']:.6f})")

        all_results[set_name] = {
            "description": description,
            "n_samples":   int(n),
            "models":      set_results,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "ood_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
