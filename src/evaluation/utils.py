"""
evaluation/utils.py — Shared inference and metric utilities for all eval scripts.
"""

import os
import numpy as np
import torch


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cls, latent_dim, ckpt_path, device):
    """Instantiate model, load weights. Returns None if checkpoint missing."""
    if not os.path.exists(ckpt_path):
        return None
    model = cls(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def load_registry(registry, device):
    """
    Load all available models from a registry list.

    Args:
        registry : list of (display_name, cls, latent_dim, ckpt_path)
    Returns:
        dict {display_name: model}  — only entries where checkpoint exists
    """
    models = {}
    for name, cls, latent_dim, path in registry:
        m = load_model(cls, latent_dim, path, device)
        if m is not None:
            models[name] = m
        else:
            print(f"  [skip] {name} — checkpoint not found: {path}")
    return models


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_outputs(model, loader, device):
    """
    Run model over a DataLoader in eval mode.
    Returns (x, y, x_hat, z) — CPU tensors concatenated over the full loader.

    z is always (N, 2*latent_dim, T) regardless of model type.
    """
    all_x, all_y, all_xhat, all_z = [], [], [], []
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if getattr(model, 'physics_type', None) is not None:
            x_hat, q, p, _ = model(x)
            z = torch.cat([q, p], dim=1)
        else:
            x_hat, z = model(x)
        all_x.append(x.cpu())
        all_y.append(y.cpu())
        all_xhat.append(x_hat.cpu())
        all_z.append(z.cpu())
    return (torch.cat(all_x), torch.cat(all_y),
            torch.cat(all_xhat), torch.cat(all_z))


@torch.no_grad()
def infer_batch(model, x_tensor, device, batch_size=64):
    """Run model on a plain tensor (no labels), return x_hat (CPU)."""
    model.eval()
    outputs = []
    for i in range(0, x_tensor.shape[0], batch_size):
        xb = x_tensor[i:i + batch_size].to(device)
        outputs.append(model(xb)[0].cpu())
    return torch.cat(outputs)


# ── Scalar metrics ─────────────────────────────────────────────────────────────

def mse(a, b):
    return torch.mean((a - b) ** 2).item()


def overlap(x_hat, y):
    """Mean cosine similarity across samples."""
    xh = x_hat.reshape(x_hat.shape[0], -1)
    yt = y.reshape(y.shape[0], -1)
    dot  = (xh * yt).sum(dim=1)
    norm = torch.clamp(xh.norm(dim=1) * yt.norm(dim=1), min=1e-12)
    return (dot / norm).mean().item()


# ── Latent structure ──────────────────────────────────────────────────────────

def _pc1_ratio_single(z_sample: np.ndarray) -> float:
    """Fraction of variance explained by PC1 of a single (C, T) latent trajectory."""
    X = z_sample.T - z_sample.T.mean(axis=0)
    if X.shape[1] == 1:
        return 1.0
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    var = s ** 2
    return float(var[0] / var.sum()) if var.sum() > 1e-30 else 0.0


def compute_pc1_ratio(model, loader, device) -> float:
    """Average PC1 variance ratio across all samples in loader."""
    _, _, _, z_all = collect_outputs(model, loader, device)
    ratios = [_pc1_ratio_single(z_all[i].numpy()) for i in range(z_all.shape[0])]
    return float(np.mean(ratios))


def compute_metrics(model, loader, device):
    """MSE, overlap, and PC1 ratio for one model on one loader."""
    _, y, x_hat, _ = collect_outputs(model, loader, device)
    return {
        "mse":     mse(x_hat, y),
        "overlap": overlap(x_hat, y),
        "pc1":     compute_pc1_ratio(model, loader, device),
    }
