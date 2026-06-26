"""
hparam_physics.py — Hyperparameter search for second-order physics models.

Searches lr × latent_dim for ProperHAE at a fixed lambda.
The lr grid is skewed lower than the HAE search because second-order autograd
(torch.autograd.grad through the Hamiltonian MLP) produces noisier gradients.

The best (latent_dim, lr) is written to BEST_CFG_PATH under "properhae" and
"porthae" (PortHAE shares the config since it has the same gradient structure).

Usage:
  python src/training/hparam_physics.py
  python src/training/hparam_physics.py --hdf ./output/dataset.hdf
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
from core.models import ProperHAE
from core.train  import quick_train
from training.config import (
    HDF_PATH, RESULTS_DIR, BEST_CFG_PATH, SEED, BATCH_SIZE,
    LATENT_DIM_GRID, PHYSICS_LR_GRID,
    HPARAM_EPOCHS, HPARAM_PATIENCE, HPARAM_LAMBDA_FIXED,
    DEFAULT_PHYSICS_CFG,
)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_hparam_physics(loaders, device, results_dir=RESULTS_DIR, seed=SEED):
    """
    Grid search over ProperHAE: latent_dim × lr at fixed lambda=HPARAM_LAMBDA_FIXED.
    Returns best config dict {"latent_dim": int, "lr": float}.
    Writes result to best_configs.json under "properhae" and "porthae".
    """
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "hparam_physics.csv")

    configs = [
        {"latent_dim": ld, "lr": lr}
        for ld in LATENT_DIM_GRID
        for lr in PHYSICS_LR_GRID
    ]

    print(f"\nPhysics hparam search (ProperHAE) — {len(configs)} configs "
          f"(latent_dim∈{LATENT_DIM_GRID}, lr∈{PHYSICS_LR_GRID}, "
          f"λ fixed={HPARAM_LAMBDA_FIXED})")
    print("Result will also be applied to PortHAE (same gradient structure).")

    rows = []
    for cfg in tqdm(configs, desc="ProperHAE hparam"):
        _set_seed(seed)
        model = ProperHAE(latent_dim=cfg["latent_dim"]).to(device)
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
    print(f"\nBest physics config: {best_cfg}  (val_mse={best_row['val_mse']:.6f})")
    print(f"Applied to both ProperHAE and PortHAE.")

    cfg_path = os.path.join(results_dir, os.path.basename(BEST_CFG_PATH))
    existing = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            existing = json.load(f)
    existing["properhae"] = best_cfg
    existing["porthae"]   = best_cfg
    with open(cfg_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved to {cfg_path}")

    return best_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf",         default=HDF_PATH)
    ap.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, _ = make_splits(args.hdf, batch_size=args.batch_size)
    run_hparam_physics(loaders, device, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
