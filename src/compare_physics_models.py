"""
compare_physics_models.py — Compare Proper/Port H-AE variants against variance HAE and CAE.

Usage:
  python src/compare_physics_models.py --hdf ./output/dataset.hdf --batch-size 16

Saves results to ./results/compare_physics_results.json
"""

import os
import json
import argparse

import torch

from config import get_device, set_seeds, HDF_PATH, BATCH_SIZE, CHECKPOINT_DIR
from data import make_splits
from models import HAE, BaselineCAE, ProperHAE, PortHAE


def collect_outputs(model, loader, device):
    """Run model over loader and return (x, y, x_hat, z) as CPU tensors."""
    model.eval()
    all_x, all_y, all_xhat, all_z = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # support both (x_hat, z) and (x_hat, q, p, H_t)
            if isinstance(out, tuple) and len(out) == 4:
                x_hat, q, p, _ = out
                z = torch.cat([q, p], dim=1)
            else:
                x_hat, z = out
            all_x.append(x.cpu())
            all_y.append(y.cpu())
            all_xhat.append(x_hat.cpu())
            all_z.append(z.cpu())
    return torch.cat(all_x), torch.cat(all_y), torch.cat(all_xhat), torch.cat(all_z)


def mse(a, b):
    return float(((a - b) ** 2).mean())


def overlap(x_hat, y):
    xh = x_hat.reshape(x_hat.shape[0], -1)
    yt = y.reshape(y.shape[0], -1)
    dot = (xh * yt).sum(dim=1)
    norm = torch.clamp(xh.norm(dim=1) * yt.norm(dim=1), min=1e-12)
    return float((dot / norm).mean())


def compute_pc1_ratio(z_all):
    # z_all: (N, C, T) — compute per-sample PC1 ratio (numpy)
    import numpy as np
    def _pc1(z_sample):
        X = z_sample.T
        X = X - X.mean(axis=0)
        if X.shape[1] == 1:
            return 1.0
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        var = s ** 2
        return float(var[0] / var.sum()) if var.sum() > 1e-30 else 0.0

    ratios = [_pc1(z_all[i].numpy()) for i in range(z_all.shape[0])]
    return float(sum(ratios) / len(ratios)) if ratios else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf', default=HDF_PATH)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--checkpoint-dir', default=CHECKPOINT_DIR)
    parser.add_argument('--results', default='./results/compare_physics_results.json')
    parser.add_argument('--lambdas', nargs='+', type=float, default=[0.1, 0.5, 1.0])
    args = parser.parse_args()

    set_seeds(42)
    device = get_device()

    print(f"Loading data from: {args.hdf}")
    loaders, info = make_splits(args.hdf, batch_size=args.batch_size)

    # load best config to get latent_dim for instantiating models
    best_cfg_path = './results/best_configs.json'
    if os.path.exists(best_cfg_path):
        with open(best_cfg_path) as f:
            cfgs = json.load(f)
        latent_dim = int(cfgs.get('hae', {}).get('latent_dim', 2))
    else:
        latent_dim = 2

    results = {}

    # baseline trained models expected names
    model_files = [
        ('hae', HAE, os.path.join(args.checkpoint_dir, 'hae_best.pt')),
        ('cae', BaselineCAE, os.path.join(args.checkpoint_dir, 'cae_best.pt')),
    ]

    # add proper/port variants for each lambda
    for lam in args.lambdas:
        model_files.append((f'properhae_lambda{lam}', ProperHAE, os.path.join(args.checkpoint_dir, f'properhae_lambda{lam}_best.pt')))
        model_files.append((f'porthae_lambda{lam}', PortHAE, os.path.join(args.checkpoint_dir, f'porthae_lambda{lam}_best.pt')))

    for name, cls, path in model_files:
        print(f"\nEvaluating: {name}")
        model = cls(latent_dim=latent_dim).to(device)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"  Loaded checkpoint: {path}")
        else:
            print(f"  Checkpoint not found at {path} — using random weights.")

        x, y, xhat, z = collect_outputs(model, loaders['test'], device)
        m = mse(xhat, y)
        ov = overlap(xhat, y)
        pc1 = compute_pc1_ratio(z)
        print(f"  test_mse={m:.6f}  overlap={ov:.4f}  pc1={pc1:.4f}")
        results[name] = {'test_mse': m, 'overlap': ov, 'pc1_ratio': pc1}

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison results to {args.results}")


if __name__ == '__main__':
    main()
