"""
train_all.py — Train all model × lambda combinations.

Reads best_configs.json for (latent_dim, lr) per model family.
Falls back to DEFAULT_* configs if the file is missing.

Checkpoint layout:
  checkpoints/hae/lambda{x}_best.pt
  checkpoints/cae/best.pt
  checkpoints/properhae/lambda{x}_best.pt
  checkpoints/porthae/lambda{x}_best.pt

Models trained:
  HAE       × LAMBDA_VALUES  (4 checkpoints)
  CAE       × 1              (1 checkpoint, lambda is irrelevant)
  ProperHAE × LAMBDA_VALUES  (4 checkpoints)
  PortHAE   × LAMBDA_VALUES  (4 checkpoints)

Total: 13 checkpoints.

Usage:
  python src/training/train_all.py
  python src/training/train_all.py --hdf ./output/dataset.hdf --batch-size 16
  python src/training/train_all.py --models hae cae          # subset
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse

import numpy as np
import torch

from core.data   import make_splits
from core.models import HAE, BaselineCAE, ProperHAE, PortHAE
from core.train  import train_model
from training.config import (
    HDF_PATH, CHECKPOINT_DIR, RESULTS_DIR, LOG_PATH, BEST_CFG_PATH,
    SEED, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LAMBDA_VALUES,
    DEFAULT_HAE_CFG, DEFAULT_PHYSICS_CFG,
)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_best_configs(cfg_path):
    """Load best_configs.json, fall back to defaults for missing keys."""
    cfgs = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfgs = json.load(f)
        print(f"Loaded best configs from {cfg_path}")
    else:
        print(f"best_configs.json not found at {cfg_path} — using defaults.")

    hae_cfg     = cfgs.get("hae",       DEFAULT_HAE_CFG)
    cae_cfg     = cfgs.get("cae",       DEFAULT_HAE_CFG)
    physics_cfg = cfgs.get("properhae", DEFAULT_PHYSICS_CFG)

    print(f"  HAE / CAE   : latent_dim={hae_cfg['latent_dim']}  lr={hae_cfg['lr']:.0e}")
    print(f"  ProperHAE/Port: latent_dim={physics_cfg['latent_dim']}  lr={physics_cfg['lr']:.0e}")
    return hae_cfg, cae_cfg, physics_cfg


def train_all(loaders, device,
              checkpoint_dir=CHECKPOINT_DIR,
              results_dir=RESULTS_DIR,
              log_path=LOG_PATH,
              cfg_path=BEST_CFG_PATH,
              lambda_values=LAMBDA_VALUES,
              max_epochs=MAX_EPOCHS,
              patience=PATIENCE,
              seed=SEED,
              models_to_train=None):
    """
    Train all model × lambda combinations.

    Args:
        models_to_train : list of model keys to train, e.g. ["hae", "cae"].
                          None = train all four families.
    Returns:
        histories : dict mapping checkpoint name → training history
    """
    if models_to_train is None:
        models_to_train = ["hae", "cae", "properhae", "porthae"]

    hae_cfg, cae_cfg, physics_cfg = _load_best_configs(cfg_path)
    os.makedirs(results_dir, exist_ok=True)
    histories = {}

    # ── HAE × all lambdas ─────────────────────────────────────────────────────
    if "hae" in models_to_train:
        ckpt_dir = os.path.join(checkpoint_dir, "hae")
        for lam in lambda_values:
            name = f"lambda{lam}"
            print(f"\n{'='*55}")
            print(f"  HAE  λ={lam}  |  ld={hae_cfg['latent_dim']}  lr={hae_cfg['lr']:.0e}")
            print(f"{'='*55}")
            _set_seed(seed)
            model = HAE(latent_dim=hae_cfg["latent_dim"]).to(device)
            hist, _ = train_model(
                model, loaders, device,
                model_name     = name,
                checkpoint_dir = ckpt_dir,
                log_path       = log_path,
                lambda_phys    = lam,
                lr             = hae_cfg["lr"],
                max_epochs     = max_epochs,
                patience       = patience,
            )
            histories[f"hae_{name}"] = hist

    # ── CAE — single run, no lambda ───────────────────────────────────────────
    if "cae" in models_to_train:
        ckpt_dir = os.path.join(checkpoint_dir, "cae")
        print(f"\n{'='*55}")
        print(f"  CAE  |  ld={cae_cfg['latent_dim']}  lr={cae_cfg['lr']:.0e}")
        print(f"{'='*55}")
        _set_seed(seed)
        model = BaselineCAE(latent_dim=cae_cfg["latent_dim"]).to(device)
        hist, _ = train_model(
            model, loaders, device,
            model_name     = "best",
            checkpoint_dir = ckpt_dir,
            log_path       = log_path,
            lambda_phys    = 0.0,
            lr             = cae_cfg["lr"],
            max_epochs     = max_epochs,
            patience       = patience,
        )
        histories["cae"] = hist

    # ── ProperHAE × all lambdas ───────────────────────────────────────────────
    if "properhae" in models_to_train:
        ckpt_dir = os.path.join(checkpoint_dir, "properhae")
        for lam in lambda_values:
            name = f"lambda{lam}"
            print(f"\n{'='*55}")
            print(f"  ProperHAE  λ={lam}  |  ld={physics_cfg['latent_dim']}  "
                  f"lr={physics_cfg['lr']:.0e}")
            print(f"{'='*55}")
            _set_seed(seed)
            model = ProperHAE(latent_dim=physics_cfg["latent_dim"]).to(device)
            hist, _ = train_model(
                model, loaders, device,
                model_name     = name,
                checkpoint_dir = ckpt_dir,
                log_path       = log_path,
                lambda_phys    = lam,
                lr             = physics_cfg["lr"],
                max_epochs     = max_epochs,
                patience       = patience,
            )
            histories[f"properhae_{name}"] = hist

    # ── PortHAE × all lambdas ─────────────────────────────────────────────────
    if "porthae" in models_to_train:
        ckpt_dir = os.path.join(checkpoint_dir, "porthae")
        for lam in lambda_values:
            name = f"lambda{lam}"
            print(f"\n{'='*55}")
            print(f"  PortHAE  λ={lam}  |  ld={physics_cfg['latent_dim']}  "
                  f"lr={physics_cfg['lr']:.0e}")
            print(f"{'='*55}")
            _set_seed(seed)
            model = PortHAE(latent_dim=physics_cfg["latent_dim"]).to(device)
            hist, _ = train_model(
                model, loaders, device,
                model_name     = name,
                checkpoint_dir = ckpt_dir,
                log_path       = log_path,
                lambda_phys    = lam,
                lr             = physics_cfg["lr"],
                max_epochs     = max_epochs,
                patience       = patience,
            )
            histories[f"porthae_{name}"] = hist

    # Save histories for loss curve plots
    histories_path = os.path.join(results_dir, "training_histories.json")
    with open(histories_path, "w") as f:
        json.dump(histories, f, indent=2)
    print(f"\nTraining histories saved to {histories_path}")

    return histories


def main():
    ap = argparse.ArgumentParser(description="Train all GW autoencoder models.")
    ap.add_argument("--hdf",          default=HDF_PATH)
    ap.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--results-dir",  default=RESULTS_DIR)
    ap.add_argument("--max-epochs",   type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience",     type=int, default=PATIENCE)
    ap.add_argument("--models",       nargs="+",
                    choices=["hae", "cae", "properhae", "porthae"],
                    default=None,
                    help="Subset of models to train. Default: all four.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, info = make_splits(args.hdf, batch_size=args.batch_size)

    train_all(
        loaders,
        device,
        checkpoint_dir  = args.checkpoint_dir,
        results_dir     = args.results_dir,
        log_path        = os.path.join(args.results_dir, "training_log.txt"),
        cfg_path        = os.path.join(args.results_dir, "best_configs.json"),
        max_epochs      = args.max_epochs,
        patience        = args.patience,
        models_to_train = args.models,
    )


if __name__ == "__main__":
    main()
