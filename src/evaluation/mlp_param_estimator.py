"""
mlp_param_estimator.py — Parameter recovery from frozen latent representations.

Trains a small MLP (encoder frozen) to recover [mass1, mass2, spin1z, spin2z]
from the latent space of each available model. Evaluates on both OOD test sets.

Architecture: flatten(2·latent_dim·112) → dim//2 → ReLU → dim//4 → ReLU → 4

Usage:
  python src/evaluation/mlp_param_estimator.py
  python src/evaluation/mlp_param_estimator.py --epochs 300
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.data         import make_splits
from evaluation.utils  import load_registry, collect_outputs
from evaluation.config import (
    HDF_PATH, CHECKPOINT_DIR, RESULTS_DIR,
    BATCH_SIZE, build_model_registry,
)

PARAM_KEYS   = ["mass1", "mass2", "spin1z", "spin2z"]
MLP_LR       = 1e-3
MLP_EPOCHS   = 300
MLP_PATIENCE = 30


class ParamMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,       input_dim // 2), nn.ReLU(),
            nn.Linear(input_dim // 2,  input_dim // 4), nn.ReLU(),
            nn.Linear(input_dim // 4,  4),
        )

    def forward(self, z):
        return self.net(z.flatten(start_dim=1))


def _load_labels(hdf_path, indices):
    with h5py.File(hdf_path, "r") as f:
        cols = [f[f"injection_parameters/{k}"][indices].astype(np.float32)
                for k in PARAM_KEYS]
    return torch.from_numpy(np.stack(cols, axis=1))


def _extract_latents(encoder, loader, device):
    _, _, _, z = collect_outputs(encoder, loader, device)
    return z


def _train_mlp(mlp, z_train, y_train_norm, z_val, y_val_norm, device,
               max_epochs=MLP_EPOCHS, patience=MLP_PATIENCE):
    optimizer    = torch.optim.Adam(mlp.parameters(), lr=MLP_LR)
    loss_fn      = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(z_train, y_train_norm),
                              batch_size=64, shuffle=True)
    best_val     = float("inf")
    no_improve   = 0
    best_state   = None

    for _ in range(max_epochs):
        mlp.train()
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(mlp(zb), yb).backward()
            optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_loss = loss_fn(mlp(z_val.to(device)), y_val_norm.to(device)).item()

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        mlp.load_state_dict(best_state)
    return best_val


def _eval_mlp(mlp, z, y_true, device, y_mean, y_std):
    mlp.eval()
    with torch.no_grad():
        pred = mlp(z.to(device)).cpu() * y_std + y_mean
    mse  = ((pred - y_true) ** 2).mean(dim=0)
    return {k: {"mse": float(mse[i]), "rmse": float(mse[i].sqrt())}
            for i, k in enumerate(PARAM_KEYS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf",            default=HDF_PATH)
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--results-dir",    default=RESULTS_DIR)
    ap.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs",         type=int, default=MLP_EPOCHS)
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, info = make_splits(args.hdf, batch_size=args.batch_size)

    y_train = _load_labels(args.hdf, info["train_idx"])
    y_val   = _load_labels(args.hdf, info["val_idx"])
    y_test1 = _load_labels(args.hdf, info["test1_idx"])
    y_test2 = _load_labels(args.hdf, info["test2_idx"])

    y_mean = y_train.mean(dim=0)
    y_std  = y_train.std(dim=0).clamp(min=1e-8)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm   = (y_val   - y_mean) / y_std

    registry = build_model_registry(args.checkpoint_dir, args.results_dir)
    models   = load_registry(registry, device)
    print(f"\nLoaded {len(models)} encoder models.\n")

    all_results = {}
    for name, encoder in models.items():
        # Freeze encoder
        for p in encoder.parameters():
            p.requires_grad_(False)

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        z_train = _extract_latents(encoder, loaders["train_eval"], device)
        z_val   = _extract_latents(encoder, loaders["val"],        device)
        z_test1 = _extract_latents(encoder, loaders["test1"],      device)
        z_test2 = _extract_latents(encoder, loaders["test2"],      device)

        input_dim = z_train.shape[1] * z_train.shape[2]
        print(f"  Latent shape: {tuple(z_train.shape[1:])}  →  flat {input_dim}")

        mlp      = ParamMLP(input_dim).to(device)
        best_val = _train_mlp(mlp, z_train, y_train_norm, z_val, y_val_norm,
                              device, max_epochs=args.epochs)
        print(f"  Best val MSE (normalised): {best_val:.6f}")

        t1 = _eval_mlp(mlp, z_test1, y_test1, device, y_mean, y_std)
        t2 = _eval_mlp(mlp, z_test2, y_test2, device, y_mean, y_std)

        print(f"\n  {'Param':<10}  {'Test1 RMSE':>12}  {'Test2 RMSE':>12}")
        print(f"  {'-'*38}")
        for k in PARAM_KEYS:
            print(f"  {k:<10}  {t1[k]['rmse']:>12.4f}  {t2[k]['rmse']:>12.4f}")

        all_results[name] = {"test1": t1, "test2": t2}

    out_path = os.path.join(args.results_dir, "mlp_param_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
