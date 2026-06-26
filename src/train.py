"""
train.py — Training loop, checkpointing, and logging.

Exports:
  train_model()  — full training run with early stopping
  quick_train()  — short run for hparam search (max_epochs capped externally)
"""

import os
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from losses import reconstruction_loss, total_hdae_loss
from models import HAE, BaselineCAE, proper_hnn_loss, port_hamiltonian_loss


# ──────────────────────────────────────────────────────────────────────────────
# Single-epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, device, lambda_phys=0.0):
    """Run one training epoch. Returns dict of mean losses."""
    model.train()
    total_recon = total_phys = total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        # Physics-aware models return (x_hat, q, p, H_t)
        if isinstance(out, tuple) and len(out) == 4:
            x_hat, q, p, H_t = out
            recon = reconstruction_loss(x_hat, y)

            if model.physics_type == 'variance':
                loss, recon, phys = total_hdae_loss(x_hat, y, H_t, lambda_phys)
            elif model.physics_type == 'proper':
                z = torch.cat([q, p], dim=1)
                phys, _ = proper_hnn_loss(model.hamiltonian, z, create_graph=True)
                loss = recon + lambda_phys * phys
            elif model.physics_type == 'port':
                z = torch.cat([q, p], dim=1)
                R = model.get_R()
                phys, _ = port_hamiltonian_loss(model.hamiltonian, R, z, create_graph=True)
                loss = recon + lambda_phys * phys
            else:
                loss, recon, phys = total_hdae_loss(x_hat, y, H_t, lambda_phys)
        else:
            x_hat, _ = out
            recon = reconstruction_loss(x_hat, y)
            loss = recon
            phys = torch.tensor(0.0)

        loss.backward()
        # Gradient clipping keeps training stable for small datasets
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_recon += recon.item()
        total_phys  += phys.item()
        total_loss  += loss.item()
        n_batches   += 1

    return {
        "loss":  total_loss  / n_batches,
        "recon": total_recon / n_batches,
        "phys":  total_phys  / n_batches,
    }


def _val_epoch(model, loader, device, lambda_phys=0.0):
    """Validation pass. Returns dict of mean losses."""
    model.eval()
    total_recon = total_phys = total_loss = 0.0
    n_batches = 0

    # proper/port need autograd to compute torch.autograd.grad(H, z);
    # variance and CAE do not — avoid building unnecessary graphs for them.
    physics_type  = getattr(model, 'physics_type', None)
    needs_autograd = physics_type in ('proper', 'port')

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if needs_autograd:
            out = model(x)
        else:
            with torch.no_grad():
                out = model(x)

        if isinstance(out, tuple) and len(out) == 4:
            x_hat, q, p, H_t = out

            if physics_type == 'variance':
                with torch.no_grad():
                    loss, recon, phys = total_hdae_loss(x_hat, y, H_t, lambda_phys)
            elif physics_type == 'proper':
                with torch.no_grad():
                    recon = reconstruction_loss(x_hat, y)
                z = torch.cat([q, p], dim=1)
                with torch.enable_grad():
                    phys, _ = proper_hnn_loss(model.hamiltonian, z, create_graph=False)
                loss = recon + lambda_phys * phys
            elif physics_type == 'port':
                with torch.no_grad():
                    recon = reconstruction_loss(x_hat, y)
                z = torch.cat([q, p], dim=1)
                R = model.get_R()
                with torch.enable_grad():
                    phys, _ = port_hamiltonian_loss(model.hamiltonian, R, z, create_graph=False)
                loss = recon + lambda_phys * phys
            else:
                with torch.no_grad():
                    loss, recon, phys = total_hdae_loss(x_hat, y, H_t, lambda_phys)
        else:
            x_hat, _ = out
            with torch.no_grad():
                recon = reconstruction_loss(x_hat, y)
            loss = recon
            phys = torch.tensor(0.0)

        total_recon += recon.item()
        total_phys  += phys.item()
        total_loss  += loss.item()
        n_batches   += 1

    return {
        "loss":  total_loss  / n_batches,
        "recon": total_recon / n_batches,
        "phys":  total_phys  / n_batches,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full training run
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    loaders,
    device,
    model_name: str,
    checkpoint_dir: str = "./checkpoints",
    log_path: str       = "./results/training_log.txt",
    lambda_phys: float  = 0.0,
    lr: float           = 1e-3,
    max_epochs: int     = 200,
    patience: int       = 20,
    verbose: bool       = True,
):
    """
    Train model with Adam + early stopping on val MSE.

    Args:
        model         : HAE or BaselineCAE instance (already on device)
        loaders       : dict with "train" and "val" DataLoaders
        device        : torch.device
        model_name    : used for checkpoint filename and log tags
        checkpoint_dir: directory to save best weights
        log_path      : append training log here
        lambda_phys   : physics loss weight (ignored for CAE)
        lr            : Adam learning rate
        max_epochs    : hard epoch limit
        patience      : early-stopping patience (epochs without val improvement)
        verbose       : print progress

    Returns:
        history : dict with "train_recon", "val_recon", "train_phys" lists (one entry/epoch)
        best_val_mse : float
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_mse = math.inf
    epochs_no_improve = 0
    best_ckpt = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    history = {"train_recon": [], "val_recon": [], "train_phys": [], "val_phys": []}

    def _log(msg):
        if verbose:
            print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    _log(f"\n{'='*60}")
    _log(f"Training: {model_name}  |  lr={lr}  |  lambda_phys={lambda_phys}")
    _log(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        tr = _train_epoch(model, loaders["train"], optimizer, device, lambda_phys)
        vl = _val_epoch(model,  loaders["val"],   device, lambda_phys)

        history["train_recon"].append(tr["recon"])
        history["val_recon"].append(vl["recon"])
        history["train_phys"].append(tr["phys"])
        history["val_phys"].append(vl["phys"])

        # Early stopping on val reconstruction MSE
        val_mse = vl["recon"]
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            _log(
                f"  Epoch {epoch:4d}/{max_epochs} | "
                f"tr_recon={tr['recon']:.6f} | vl_recon={vl['recon']:.6f} | "
                f"phys={tr['phys']:.6f} | best_val={best_val_mse:.6f} | "
                f"t={elapsed:.0f}s"
            )

        if epochs_no_improve >= patience:
            _log(f"  Early stopping at epoch {epoch} (patience={patience}).")
            break

    elapsed = time.time() - t0
    _log(f"Done. Best val MSE = {best_val_mse:.6f} | Total time: {elapsed:.1f}s")
    _log(f"Checkpoint saved to: {best_ckpt}")

    return history, best_val_mse


# ──────────────────────────────────────────────────────────────────────────────
# Thin wrapper for hparam search (same loop, just with a lower max_epochs cap)
# ──────────────────────────────────────────────────────────────────────────────

def quick_train(model, loaders, device, lambda_phys=0.0, lr=1e-3,
                max_epochs=50, patience=10):
    """
    Fast training run for hyperparameter search.
    No checkpointing, no logging — just returns best val MSE.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_mse = math.inf
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        _train_epoch(model, loaders["train"], optimizer, device, lambda_phys)
        vl = _val_epoch(model, loaders["val"], device, lambda_phys)
        val_mse = vl["recon"]

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_val_mse
