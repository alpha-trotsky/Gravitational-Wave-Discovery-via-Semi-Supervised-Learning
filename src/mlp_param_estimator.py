"""
mlp_param_estimator.py — Parameter recovery from latent representations.

Trains a small MLP on (latent → [mass1, mass2, spin1z, spin2z]) for each of
the four encoder models (HAE, CAE, ProperHAE, PortHAE).

Encoder weights are frozen throughout. MLP is trained on the encoder training
set (encoder is frozen so no label leakage) with early stopping on the val set.
OOD evaluation on test1 and test2.

Architecture: Flatten(2*latent_dim*112) → D//2 → ReLU → D//4 → ReLU → 4

Usage:
  python src/mlp_param_estimator.py
  python src/mlp_param_estimator.py --hdf ./output/dataset.hdf --epochs 300
"""

import os
import json
import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (get_device, set_seeds,
                    HDF_PATH, CHECKPOINT_DIR, RESULTS_DIR, SEED, BATCH_SIZE)
from data import make_splits
from models import HAE, BaselineCAE, ProperHAE, PortHAE


PARAM_KEYS  = ["mass1", "mass2", "spin1z", "spin2z"]
MLP_LR      = 1e-3
MLP_EPOCHS  = 300
MLP_PATIENCE = 30


# ──────────────────────────────────────────────────────────────────────────────

class ParamMLP(nn.Module):
    """
    3-layer MLP: flattened latent → halve twice → 4 targets.
    Input dim D = 2 * latent_dim * 112.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,       input_dim // 2), nn.ReLU(),
            nn.Linear(input_dim // 2,  input_dim // 4), nn.ReLU(),
            nn.Linear(input_dim // 4,  4),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.flatten(start_dim=1))


# ──────────────────────────────────────────────────────────────────────────────

def extract_latent(encoder, loader, device) -> torch.Tensor:
    """Pass all batches through frozen encoder, return (N, 2*D, 112) CPU tensor."""
    encoder.eval()
    parts = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = encoder(x)
            if isinstance(out, tuple) and len(out) == 4:
                _, q, p, _ = out
                z = torch.cat([q, p], dim=1)
            else:
                _, z = out
            parts.append(z.cpu())
    return torch.cat(parts, dim=0)


def load_labels(hdf_path: str, indices: np.ndarray) -> torch.Tensor:
    """Return (N, 4) float32 tensor for [mass1, mass2, spin1z, spin2z]."""
    with h5py.File(hdf_path, "r") as f:
        cols = [f[f"injection_parameters/{k}"][indices].astype(np.float32)
                for k in PARAM_KEYS]
    return torch.from_numpy(np.stack(cols, axis=1))


# ──────────────────────────────────────────────────────────────────────────────

def train_mlp(mlp, z_train, y_train_norm,
              z_val,   y_val_norm,
              device, max_epochs=MLP_EPOCHS, patience=MLP_PATIENCE):
    """Train MLP, return best val MSE (normalised)."""
    optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR)
    loss_fn   = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(z_train, y_train_norm),
        batch_size=64, shuffle=True,
    )

    best_val   = float("inf")
    no_improve = 0
    best_state = None

    for _ in range(max_epochs):
        mlp.train()
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(mlp(zb), yb).backward()
            optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                mlp(z_val.to(device)), y_val_norm.to(device)
            ).item()

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


def eval_mlp(mlp, z, y_true, device, y_mean, y_std):
    """
    Predict on z, denormalise, return per-parameter MSE and RMSE in physical units.
    mass1/mass2 in solar masses, spins dimensionless.
    """
    mlp.eval()
    with torch.no_grad():
        pred_norm = mlp(z.to(device)).cpu()
    pred = pred_norm * y_std + y_mean

    mse  = ((pred - y_true) ** 2).mean(dim=0)
    rmse = mse.sqrt()
    return {
        k: {"mse": float(mse[i]), "rmse": float(rmse[i])}
        for i, k in enumerate(PARAM_KEYS)
    }


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf",            default=HDF_PATH)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--results-dir",    default=RESULTS_DIR)
    parser.add_argument("--epochs",         type=int, default=MLP_EPOCHS)
    args = parser.parse_args()

    set_seeds(SEED)
    device = get_device()

    print("Loading data ...")
    loaders, info = make_splits(args.hdf, batch_size=BATCH_SIZE)

    # Labels for all splits
    y_train = load_labels(args.hdf, info["train_idx"])
    y_val   = load_labels(args.hdf, info["val_idx"])
    y_test1 = load_labels(args.hdf, info["test1_idx"])
    y_test2 = load_labels(args.hdf, info["test2_idx"])

    # Normalise using training set statistics
    y_mean = y_train.mean(dim=0)
    y_std  = y_train.std(dim=0).clamp(min=1e-8)

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm   = (y_val   - y_mean) / y_std

    print(f"\nParameter ranges (train set):")
    for i, k in enumerate(PARAM_KEYS):
        print(f"  {k:<8}: mean={y_mean[i]:.3f}  std={y_std[i]:.3f}")

    # Load best configs for latent_dim
    cfg_path = os.path.join(args.results_dir, "best_configs.json")
    cfgs = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfgs = json.load(f)

    model_specs = [
        ("hae",       HAE,         int(cfgs.get("hae",       {}).get("latent_dim", 2))),
        ("cae",       BaselineCAE, int(cfgs.get("cae",       {}).get("latent_dim", 2))),
        ("properhae", ProperHAE,   int(cfgs.get("properhae", {}).get("latent_dim", 2))),
        ("porthae",   PortHAE,     int(cfgs.get("porthae",   {}).get("latent_dim", 2))),
    ]

    all_results = {}

    for name, cls, latent_dim in model_specs:
        ckpt = os.path.join(args.checkpoint_dir, f"{name}_best.pt")
        if not os.path.exists(ckpt):
            print(f"\n  Skipping {name} — checkpoint not found at {ckpt}")
            continue

        print(f"\n{'='*55}")
        print(f"  {name}  (latent_dim={latent_dim})")
        print(f"{'='*55}")

        encoder = cls(latent_dim=latent_dim).to(device)
        encoder.load_state_dict(torch.load(ckpt, map_location=device))
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        print("  Extracting latents ...")
        z_train_t = extract_latent(encoder, loaders["train_eval"], device)
        z_val_t   = extract_latent(encoder, loaders["val"],        device)
        z_test1_t = extract_latent(encoder, loaders["test1"],      device)
        z_test2_t = extract_latent(encoder, loaders["test2"],      device)

        input_dim = z_train_t.shape[1] * z_train_t.shape[2]
        print(f"  Latent shape: {tuple(z_train_t.shape[1:])}  →  flat {input_dim}")
        print(f"  MLP: {input_dim} → {input_dim//2} → {input_dim//4} → 4")

        mlp = ParamMLP(input_dim).to(device)
        best_val = train_mlp(
            mlp,
            z_train_t, y_train_norm,
            z_val_t,   y_val_norm,
            device, max_epochs=args.epochs,
        )
        print(f"  Best val MSE (normalised): {best_val:.6f}")

        test1_metrics = eval_mlp(mlp, z_test1_t, y_test1, device, y_mean, y_std)
        test2_metrics = eval_mlp(mlp, z_test2_t, y_test2, device, y_mean, y_std)

        print(f"\n  {'Param':<10} {'Test1 RMSE':>12} {'Test2 RMSE':>12}")
        print(f"  {'-'*36}")
        for k in PARAM_KEYS:
            r1 = test1_metrics[k]["rmse"]
            r2 = test2_metrics[k]["rmse"]
            print(f"  {k:<10} {r1:>12.4f} {r2:>12.4f}")

        all_results[name] = {"test1": test1_metrics, "test2": test2_metrics}

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "mlp_param_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
