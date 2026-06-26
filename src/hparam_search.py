"""
hparam_search.py — Nested-loop grid search over (lambda_phys, latent_dim, lr).

Grid (applied to HAE, ProperHAE, PortHAE):
  lambda_phys ∈ {0.01, 0.1, 1.0}
  latent_dim  ∈ {1, 2, 4}
  lr          ∈ {1e-4, 1e-3}

CAE uses the same latent_dim / lr grid without a lambda dimension.

Each config is trained for max 50 epochs with early stopping (patience=10).
Results are saved to ./results/hparam_search.csv.

Returns:
  best_hae_cfg, best_cae_cfg, best_properhae_cfg, best_porthae_cfg
"""

import os
import torch
import pandas as pd
from tqdm import tqdm

from models import HAE, BaselineCAE, ProperHAE, PortHAE
from train  import quick_train


LAMBDA_PHYS_GRID  = [0.01, 0.1, 1.0]
LATENT_DIM_GRID   = [1, 2, 4]
LR_GRID           = [1e-4, 1e-3]
HPARAM_PATIENCE   = 10
HPARAM_MAX_EPOCHS = 50


def _search_block(model_label, model_cls, configs, loaders, device, seed, rows,
                  has_lambda=True):
    """Run quick_train over all configs for one model class, append to rows."""
    print(f"\n── {model_label} configs ({len(configs)}) ──")
    for cfg in tqdm(configs, desc=f"{model_label} hparam search"):
        _set_seed(seed)
        model   = model_cls(latent_dim=cfg["latent_dim"]).to(device)
        lp      = cfg.get("lambda_phys", 0.0)
        val_mse = quick_train(
            model, loaders, device,
            lambda_phys = lp,
            lr          = cfg["lr"],
            max_epochs  = HPARAM_MAX_EPOCHS,
            patience    = HPARAM_PATIENCE,
        )
        row = {"model": model_label.lower().replace("-", ""),
               "lambda_phys": lp,
               "latent_dim":  cfg["latent_dim"],
               "lr":          cfg["lr"],
               "val_mse":     val_mse}
        rows.append(row)
        lp_str = f"lp={lp:.2f} " if has_lambda else ""
        tqdm.write(f"  {model_label:10s} | {lp_str}ld={cfg['latent_dim']} "
                   f"lr={cfg['lr']:.0e} → val_mse={val_mse:.6f}")


def run_hparam_search(loaders, device, results_dir="./results", seed=42):
    """
    Full grid search over HAE, CAE, ProperHAE, PortHAE.

    Returns:
        best_hae_cfg, best_cae_cfg, best_properhae_cfg, best_porthae_cfg
    """
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "hparam_search.csv")

    phys_configs = [
        {"lambda_phys": lp, "latent_dim": ld, "lr": lr}
        for lp in LAMBDA_PHYS_GRID
        for ld in LATENT_DIM_GRID
        for lr in LR_GRID
    ]
    cae_configs = [
        {"latent_dim": ld, "lr": lr}
        for ld in LATENT_DIM_GRID
        for lr in LR_GRID
    ]

    n_total = 3 * len(phys_configs) + len(cae_configs)
    print(f"\nHyperparameter search — {n_total} configs total "
          f"({len(phys_configs)} each for HAE/ProperHAE/PortHAE, "
          f"{len(cae_configs)} for CAE)")

    rows = []
    _search_block("HAE",       HAE,         phys_configs, loaders, device, seed, rows)
    _search_block("CAE",       BaselineCAE, cae_configs,  loaders, device, seed, rows,
                  has_lambda=False)
    _search_block("ProperHAE", ProperHAE,   phys_configs, loaders, device, seed, rows)
    _search_block("PortHAE",   PortHAE,     phys_configs, loaders, device, seed, rows)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    def _best(model_key, include_lambda=True):
        sub = df[df["model"] == model_key]
        row = sub.loc[sub["val_mse"].idxmin()]
        cfg = {"latent_dim": int(row["latent_dim"]), "lr": float(row["lr"])}
        if include_lambda:
            cfg["lambda_phys"] = float(row["lambda_phys"])
        print(f"  Best {model_key:10s}: {cfg}  (val_mse={row['val_mse']:.6f})")
        return cfg

    print("\nBest configs:")
    best_hae_cfg       = _best("hae")
    best_cae_cfg       = _best("cae", include_lambda=False)
    best_properhae_cfg = _best("properhae")
    best_porthae_cfg   = _best("porthae")

    return best_hae_cfg, best_cae_cfg, best_properhae_cfg, best_porthae_cfg


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
