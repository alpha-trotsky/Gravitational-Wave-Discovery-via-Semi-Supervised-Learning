"""
run_physics_variants.py — Train ProperHAE and PortHAE across specified λ values.

Usage (example):
  python src/run_physics_variants.py --hdf ./output/dataset.hdf --batch-size 16

This script reads ./results/best_configs.json for best `latent_dim` and `lr`.
It saves checkpoints to the usual `./checkpoints` folder with descriptive names.
"""

import os
import json
import argparse
import torch

from config import get_device, set_seeds, HDF_PATH, BATCH_SIZE, MAX_EPOCHS, PATIENCE, CHECKPOINT_DIR, LOG_PATH, PLOTS_DIR
from data import make_splits
from models import ProperHAE, PortHAE
from train import train_model
from evaluate import plot_physics_variant_curves, plot_physics_variant_phase_space

LAMBDA_GRID = [0.1, 0.5, 1.0]
SEED = 42


def load_best_cfg(path="./results/best_configs.json"):
    """Return (properhae_cfg, porthae_cfg) falling back to HAE config or defaults."""
    defaults = {"latent_dim": 2, "lr": 1e-3, "lambda_phys": 0.5}
    if os.path.exists(path):
        with open(path) as f:
            cfgs = json.load(f)
        properhae_cfg = cfgs.get("properhae") or cfgs.get("hae") or defaults
        porthae_cfg   = cfgs.get("porthae")   or cfgs.get("hae") or defaults
        return properhae_cfg, porthae_cfg
    return defaults.copy(), defaults.copy()


def run_physics_variants(loaders, device, latent_dim, lr,
                         checkpoint_dir=CHECKPOINT_DIR,
                         log_path=LOG_PATH,
                         plots_dir=PLOTS_DIR,
                         max_epochs=MAX_EPOCHS,
                         patience=PATIENCE,
                         cae=None):
    """
    Train ProperHAE and PortHAE across LAMBDA_GRID.

    After training, loads best checkpoints and generates:
      - physics_variant_loss_curves.png  (train/val curves per model)
      - physics_phase_space_proper.png   (latent trajectories, ProperHAE)
      - physics_phase_space_port.png     (latent trajectories, PortHAE)

    Args:
        cae : optional loaded BaselineCAE used as reference in phase-space plots.

    Returns: histories dict keyed by model name.
    """
    experiments = [
        (ProperHAE, "properhae"),
        (PortHAE,   "porthae"),
    ]

    histories     = {}
    trained_models = {}

    for ModelCls, shortname in experiments:
        for lam in LAMBDA_GRID:
            set_seeds(SEED)
            model      = ModelCls(latent_dim=latent_dim).to(device)
            model_name = f"{shortname}_lambda{lam}"
            print(f"\n--- Training {ModelCls.__name__} | lambda={lam} | "
                  f"ld={latent_dim} lr={lr} ---")
            hist, _ = train_model(
                model, loaders, device,
                model_name     = model_name,
                checkpoint_dir = checkpoint_dir,
                log_path       = log_path,
                lambda_phys    = lam,
                lr             = lr,
                max_epochs     = max_epochs,
                patience       = patience,
            )
            histories[model_name] = hist

            # Load best checkpoint weights for plotting
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            trained_models[model_name] = model

    plot_physics_variant_curves(histories, plots_dir)

    # Use whichever test split is available (new: test1, legacy: test)
    test_loader = loaders.get("test1") or loaders.get("test")
    if test_loader is not None:
        plot_physics_variant_phase_space(
            trained_models, test_loader, device, plots_dir, cae=cae
        )
    else:
        print("  No test loader found — skipping phase space plots.")

    return histories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf",        default=HDF_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience",   type=int, default=PATIENCE)
    args = parser.parse_args()

    set_seeds(SEED)
    device = get_device()

    print(f"Loading data from: {args.hdf}")
    loaders, _ = make_splits(args.hdf, batch_size=args.batch_size)

    properhae_cfg, porthae_cfg = load_best_cfg()

    # Lambda sweep uses the best lr/latent_dim from hparam search per model type.
    # Lambda itself is swept across LAMBDA_GRID for analysis.
    run_physics_variants(loaders, device,
                         latent_dim = int(properhae_cfg.get("latent_dim", 2)),
                         lr         = float(properhae_cfg.get("lr", 1e-3)),
                         max_epochs = args.max_epochs,
                         patience   = args.patience)
    print("All experiments finished.")


if __name__ == '__main__':
    main()
