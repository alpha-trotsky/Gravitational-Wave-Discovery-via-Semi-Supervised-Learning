"""
hparam_hae.py — Hyperparameter search for the variance-based HAE.

Searches lr × latent_dim at a fixed lambda (HPARAM_LAMBDA_FIXED).
The best (latent_dim, lr) is written to BEST_CFG_PATH and applied to both
HAE (all lambda sweeps) and CAE (single training run).

Usage:
  python src/training/hparam_hae.py
  python src/training/hparam_hae.py --hdf ./output/dataset.hdf --batch-size 16
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.data   import make_splits
from core.models import HAE
from core.train  import quick_train
from training.config import (
    HDF_PATH, RESULTS_DIR, BEST_CFG_PATH, SEED, BATCH_SIZE,
    LATENT_DIM_GRID, HAE_LR_GRID,
    HPARAM_EPOCHS, HPARAM_PATIENCE, HPARAM_LAMBDA_FIXED,
    DEFAULT_HAE_CFG,
)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_hparam_hae(loaders, device, results_dir=RESULTS_DIR, seed=SEED):
    """
    Grid search over HAE: latent_dim × lr at fixed lambda=HPARAM_LAMBDA_FIXED.
    Returns best config dict {"latent_dim": int, "lr": float}.
    Writes result to best_configs.json (merged with any existing entries).
    """
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "hparam_hae.csv")

    configs = [
        {"latent_dim": ld, "lr": lr}
        for ld in LATENT_DIM_GRID
        for lr in HAE_LR_GRID
    ]

    print(f"\nHAE hparam search — {len(configs)} configs "
          f"(latent_dim∈{LATENT_DIM_GRID}, lr∈{HAE_LR_GRID}, "
          f"λ fixed={HPARAM_LAMBDA_FIXED})")

    rows = []
    for cfg in tqdm(configs, desc="HAE hparam"):
        _set_seed(seed)
        model = HAE(latent_dim=cfg["latent_dim"]).to(device)
        val_mse = quick_train(
            model, loaders, device,
            lambda_phys = HPARAM_LAMBDA_FIXED,
            lr          = cfg["lr"],
            max_epochs  = HPARAM_EPOCHS,
            patience    = HPARAM_PATIENCE,
        )
        rows.append({**cfg, "val_mse": val_mse})
        tqdm.write(f"  ld={cfg['latent_dim']}  lr={cfg['lr']:.0e}  "
                   f"→ val_mse={val_mse:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    best_row = df.loc[df["val_mse"].idxmin()]
    best_cfg = {
        "latent_dim": int(best_row["latent_dim"]),
        "lr":         float(best_row["lr"]),
    }
    print(f"\nBest HAE config: {best_cfg}  (val_mse={best_row['val_mse']:.6f})")
    print(f"Applied to CAE (same backbone, no physics loss).")

    # Merge into best_configs.json
    cfg_path = os.path.join(results_dir, os.path.basename(BEST_CFG_PATH))
    existing = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            existing = json.load(f)
    existing["hae"] = best_cfg
    existing["cae"] = best_cfg   # CAE mirrors HAE best config
    with open(cfg_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved to {cfg_path}")

    return best_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf",        default=HDF_PATH)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, _ = make_splits(args.hdf, batch_size=args.batch_size)
    run_hparam_hae(loaders, device, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
