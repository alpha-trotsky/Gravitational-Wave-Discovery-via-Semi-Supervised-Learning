"""
evaluate.py — All evaluation experiments and plot generation.

Experiments:
  1. Reconstruction quality  : test MSE, waveform overlap, PC1 ratio, 4-sample plot
  2. Latent space structure  : phase-space trajectories (H-AE vs CAE)
  3. Generalization           : train vs test MSE bar chart (held-out mass range)
  4. Loss curves             : train/val MSE per epoch, physics loss
  5. Noise robustness        : OOD evaluation with noisy h1_strain
  6. Lambda sweep            : train H-AE at λ∈{0.01,0.1,1.0}, compare vs CAE

All plots → ./plots/,  all numbers → ./results/final_metrics.json.
"""

import os
import json
import copy
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from losses import reconstruction_loss, total_hdae_loss
from models import HAE, BaselineCAE
from data   import load_noisy_test_data


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_outputs(model, loader, device):
    """
    Run model over a DataLoader in eval mode (no_grad guaranteed by decorator).
    Returns (x, y, x_hat, z) — all CPU tensors concatenated over the full loader.

    BUG-FIX NOTE: always call model.eval() here so GroupNorm running stats are
    not updated; the @torch.no_grad() decorator prevents gradient accumulation.
    Pass loaders["train_eval"] (shuffle=False) for train-set metrics to ensure
    deterministic ordering and no dropout/BN side-effects.
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
def _infer_batch(model, x_tensor, device, batch_size=64):
    """Run model on a plain tensor, return x_hat tensor (CPU)."""
    model.eval()
    outputs = []
    for i in range(0, x_tensor.shape[0], batch_size):
        xb = x_tensor[i:i + batch_size].to(device)
        xh = model(xb)[0]   # first return value is x_hat for both HAE and CAE
        outputs.append(xh.cpu())
    return torch.cat(outputs)


def _mse(a, b):
    return torch.mean((a - b) ** 2).item()


def _overlap(x_hat, y):
    """Mean cosine similarity across samples."""
    xh = x_hat.reshape(x_hat.shape[0], -1)
    yt = y.reshape(y.shape[0], -1)
    dot  = (xh * yt).sum(dim=1)
    norm = torch.clamp(xh.norm(dim=1) * yt.norm(dim=1), min=1e-12)
    return (dot / norm).mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# PC1 ratio — latent structure metric
# ──────────────────────────────────────────────────────────────────────────────

def _pc1_ratio_single(z_sample: np.ndarray) -> float:
    """
    Fraction of variance explained by PC1 of a single latent trajectory.

    Args:
        z_sample : (C, T) — C channels × T timesteps
    Returns:
        float in [0, 1]
    """
    X = z_sample.T          # (T, C) — T observations, C features
    X = X - X.mean(axis=0)  # centre
    if X.shape[1] == 1:
        return 1.0           # trivially 1D
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    var = s ** 2
    return float(var[0] / var.sum()) if var.sum() > 1e-30 else 0.0


def compute_pc1_ratio(model, loader, device) -> float:
    """
    Average PC1 variance ratio across all samples in loader.
    Higher = more 1D-like latent trajectory = more constrained manifold.
    """
    _, _, _, z_all = _collect_outputs(model, loader, device)
    # z_all: (N, C, T)
    ratios = [_pc1_ratio_single(z_all[i].numpy()) for i in range(z_all.shape[0])]
    return float(np.mean(ratios))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Reconstruction quality
# ──────────────────────────────────────────────────────────────────────────────

def eval_reconstruction(hae, cae, test_loader, device, plots_dir,
                        train_eval_loader=None):
    """
    Test MSE, overlap, PC1 ratio, and 4-sample waveform plot.

    Args:
        train_eval_loader : shuffle=False loader over train set — used for
                            train MSE so we're in full eval mode with no shuffle.
    """
    print("\n[Eval 1] Reconstruction quality ...")

    x_h, y_h, xhat_h, _ = _collect_outputs(hae, test_loader, device)
    x_c, y_c, xhat_c, _ = _collect_outputs(cae, test_loader, device)
    assert torch.allclose(y_h, y_c), "Test targets don't match — check loaders"

    test_mse_hae = _mse(xhat_h, y_h)
    test_mse_cae = _mse(xhat_c, y_c)
    overlap_hae  = _overlap(xhat_h, y_h)
    overlap_cae  = _overlap(xhat_c, y_c)

    # PC1 ratio on test set
    pc1_hae = compute_pc1_ratio(hae, test_loader, device)
    pc1_cae = compute_pc1_ratio(cae, test_loader, device)

    print(f"  Test MSE  — H-AE: {test_mse_hae:.6f} | CAE: {test_mse_cae:.6f}")
    print(f"  Overlap   — H-AE: {overlap_hae:.4f}  | CAE: {overlap_cae:.4f}")
    print(f"  PC1 ratio — H-AE: {pc1_hae:.4f}      | CAE: {pc1_cae:.4f}")

    train_mse_hae = train_mse_cae = None
    if train_eval_loader is not None:
        # BUG-FIX: use train_eval (shuffle=False, model in .eval()) for fair comparison
        _, y_tr, xhat_tr_h, _ = _collect_outputs(hae, train_eval_loader, device)
        _, _,    xhat_tr_c, _ = _collect_outputs(cae, train_eval_loader, device)
        train_mse_hae = _mse(xhat_tr_h, y_tr)
        train_mse_cae = _mse(xhat_tr_c, y_tr)
        print(f"  Train MSE — H-AE: {train_mse_hae:.6f} | CAE: {train_mse_cae:.6f}")

    # 4-sample reconstruction plot
    n_ex = min(4, x_h.shape[0])
    t    = np.arange(x_h.shape[-1])
    fig, axes = plt.subplots(n_ex, 4, figsize=(20, 4 * n_ex), sharex=True)
    if n_ex == 1:
        axes = axes[None, :]
    col_titles = ["Input (clean signal)", "Target (clean signal)",
                  "H-AE Output", "CAE Output"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")
    for i in range(n_ex):
        for col, d in enumerate([x_h[i,0], y_h[i,0], xhat_h[i,0], xhat_c[i,0]]):
            axes[i, col].plot(t, d.numpy(), lw=0.7)
            axes[i, col].set_ylabel(f"Sample {i+1}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Sample index")
    fig.suptitle("Clean-Signal Autoencoder Reconstructions — Test Set",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(plots_dir, "reconstructions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return {
        "test_mse_hae":          test_mse_hae,
        "test_mse_cae":          test_mse_cae,
        "overlap_hae":           overlap_hae,
        "overlap_cae":           overlap_cae,
        "latent_pc1_ratio_hae":  pc1_hae,
        "latent_pc1_ratio_cae":  pc1_cae,
        "train_mse_hae":         train_mse_hae,
        "train_mse_cae":         train_mse_cae,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Latent space / phase space
# ──────────────────────────────────────────────────────────────────────────────

def eval_phase_space(hae, cae, test_loader, device, plots_dir, n_samples=10):
    """Phase-space trajectories for H-AE (q₀ vs p₀) and CAE (ch0 vs ch1)."""
    print("\n[Eval 2] Phase-space trajectories ...")

    _, _, _, z_hae = _collect_outputs(hae, test_loader, device)
    _, _, _, z_cae = _collect_outputs(cae, test_loader, device)

    n_samples = min(n_samples, z_hae.shape[0])
    torch.manual_seed(0)
    idx = torch.randperm(z_hae.shape[0])[:n_samples]

    D  = hae.latent_dim
    q0 = z_hae[idx, 0, :].numpy()
    p0 = z_hae[idx, D, :].numpy()
    _plot_phase_space(q0, p0, n_samples,
                      title="H-AE Phase Space (q₀ vs p₀)",
                      path=os.path.join(plots_dir, "phase_space.png"))

    z0 = z_cae[idx, 0, :].numpy()
    z1 = z_cae[idx, min(1, z_cae.shape[1]-1), :].numpy()
    _plot_phase_space(z0, z1, n_samples,
                      title="CAE Latent Space (channel 0 vs channel 1)",
                      path=os.path.join(plots_dir, "phase_space_baseline.png"),
                      xlabel="z₀", ylabel="z₁")


def _plot_phase_space(a, b, n_samples, title, path,
                      xlabel="q₀", ylabel="p₀",
                      xlim=None, ylim=None, ax_in=None, fig_in=None,
                      show_colorbar=True):
    """
    Scatter phase-space trajectory coloured by time (plasma colormap).
    Can write to an existing axes (ax_in) for composite figures, or save
    a standalone figure to `path`.
    """
    standalone = ax_in is None

    if standalone:
        cols = min(n_samples, 5)
        rows = math.ceil(n_samples / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes_list = np.array(axes).reshape(-1) if n_samples > 1 else [axes]
    else:
        axes_list = [ax_in]
        n_samples = 1       # caller provides a single ax — draw sample 0 only

    for i in range(len(axes_list) if standalone else 1):
        T = a[i].shape[0]
        sc = axes_list[i].scatter(a[i], b[i], c=np.arange(T),
                                  cmap="plasma", s=8, rasterized=True)
        axes_list[i].set_xlabel(xlabel, fontsize=8)
        axes_list[i].set_ylabel(ylabel, fontsize=8)
        if standalone:
            axes_list[i].set_title(f"Sample {i+1}", fontsize=9)
        if xlim is not None:
            axes_list[i].set_xlim(xlim)
        if ylim is not None:
            axes_list[i].set_ylim(ylim)

    if standalone:
        for j in range(n_samples, len(axes_list)):
            axes_list[j].set_visible(False)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Generalization bar chart
# ──────────────────────────────────────────────────────────────────────────────

def eval_generalization(metrics, plots_dir):
    """Bar chart: train vs test MSE for H-AE and CAE."""
    print("\n[Eval 3] Generalization bar chart ...")

    train_mses = [metrics["train_mse_hae"], metrics["train_mse_cae"]]
    test_mses  = [metrics["test_mse_hae"],  metrics["test_mse_cae"]]
    if any(v is None for v in train_mses):
        print("  Skipping (train MSE not available).")
        return {}

    x     = np.arange(2)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    b_tr = ax.bar(x - width/2, train_mses, width,
                  label="Train MSE (mass <= 65 Msun)", color="#4C72B0")
    b_te = ax.bar(x + width/2, test_mses,  width,
                  label="Test MSE  (mass > 70 Msun)", color="#DD8452")
    for bar in list(b_tr) + list(b_te):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["H-AE", "CAE"], fontsize=12)
    ax.set_ylabel("MSE")
    ax.set_title("Generalization to Held-Out Mass Range\n"
                 "(train: both masses <= 65 Msun | test: both masses > 70 Msun)")
    ax.legend()
    ax.set_ylim(0, max(max(train_mses), max(test_mses)) * 1.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "generalization.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    gaps = {
        "generalization_gap_hae": metrics["test_mse_hae"] - metrics["train_mse_hae"],
        "generalization_gap_cae": metrics["test_mse_cae"] - metrics["train_mse_cae"],
    }
    print(f"  Gap (test−train) — H-AE: {gaps['generalization_gap_hae']:.6f} | "
          f"CAE: {gaps['generalization_gap_cae']:.6f}")
    return gaps


# ──────────────────────────────────────────────────────────────────────────────
# 4. Loss curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(hae_history, cae_history, plots_dir):
    """Train/val MSE curves for both models; physics loss curve for H-AE."""
    print("\n[Eval 4] Loss curves ...")

    if not hae_history.get("train_recon") and not cae_history.get("train_recon"):
        print("  No history available — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, hist, name in zip(axes, [hae_history, cae_history], ["H-AE", "CAE"]):
        if not hist.get("train_recon"):
            ax.set_visible(False); continue
        ep = np.arange(1, len(hist["train_recon"]) + 1)
        ax.semilogy(ep, hist["train_recon"], label="Train MSE")
        ax.semilogy(ep, hist["val_recon"],   label="Val MSE", linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log scale)")
        ax.set_title(f"{name} — Reconstruction MSE"); ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training & Validation Loss Curves", fontsize=13)
    fig.tight_layout()
    path = os.path.join(plots_dir, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    phys_tr = hae_history.get("train_phys", [])
    if phys_tr and any(v > 0 for v in phys_tr):
        fig, ax = plt.subplots(figsize=(8, 5))
        ep = np.arange(1, len(phys_tr) + 1)
        ax.semilogy(ep, phys_tr,                    label="Train physics loss")
        ax.semilogy(ep, hae_history["val_phys"], linestyle="--",
                    label="Val physics loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Hamiltonian variance (log scale)")
        ax.set_title("H-AE — Physics (Hamiltonian Conservation) Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plots_dir, "physics_loss.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Noise robustness
# ──────────────────────────────────────────────────────────────────────────────

def eval_noise_robustness(hae, cae, test_idx, hdf_path, test_loader,
                          device, plots_dir, plot_suffix=""):
    """OOD: feed whitened noisy h1_strain through models trained on clean signals."""
    print("\n[Eval 5] Noise robustness (OOD evaluation) ...")

    x_noisy = load_noisy_test_data(hdf_path, test_idx)   # (N, 1, 1792)
    print(f"  Noisy tensor: {x_noisy.shape}  "
          f"range=[{x_noisy.min():.2f},{x_noisy.max():.2f}]")

    clean_targets = [y for _, y in test_loader]
    y_clean = torch.cat(clean_targets)   # (N, 1, 1792)
    assert x_noisy.shape[0] == y_clean.shape[0]

    xhat_hae = _infer_batch(hae, x_noisy, device)
    xhat_cae = _infer_batch(cae, x_noisy, device)

    mse_hae     = _mse(xhat_hae, y_clean)
    mse_cae     = _mse(xhat_cae, y_clean)
    overlap_hae = _overlap(xhat_hae, y_clean)
    overlap_cae = _overlap(xhat_cae, y_clean)

    print(f"  MSE     — H-AE: {mse_hae:.6f} | CAE: {mse_cae:.6f}")
    print(f"  Overlap — H-AE: {overlap_hae:.4f}  | CAE: {overlap_cae:.4f}")

    n_ex = min(4, x_noisy.shape[0])
    t    = np.arange(x_noisy.shape[-1])
    fig, axes = plt.subplots(n_ex, 4, figsize=(20, 4*n_ex), sharex=True)
    if n_ex == 1:
        axes = axes[None, :]
    col_titles = ["Noisy Strain (OOD input)", "True Clean Signal",
                  "H-AE Output", "CAE Output"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")
    for i in range(n_ex):
        for col, d in enumerate([x_noisy[i,0], y_clean[i,0],
                                  xhat_hae[i,0], xhat_cae[i,0]]):
            axes[i, col].plot(t, d.numpy(), lw=0.7)
            axes[i, col].set_ylabel(f"Sample {i+1}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Sample index")
    fig.suptitle(
        "Noise Robustness — Models trained on clean signals, evaluated on noisy strain\n"
        "(OOD: noisy strain never seen during training)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(plots_dir, f"noise_robustness{plot_suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return {
        "noise_robustness_mse_hae":     mse_hae,
        "noise_robustness_mse_cae":     mse_cae,
        "noise_robustness_overlap_hae": overlap_hae,
        "noise_robustness_overlap_cae": overlap_cae,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Lambda sweep
# ──────────────────────────────────────────────────────────────────────────────

LAMBDA_SWEEP_VALUES = [0.01, 0.1, 1.0]


def run_lambda_sweep(loaders, device, latent_dim, lr,
                     checkpoint_dir, plots_dir, results_dir,
                     max_epochs=200, patience=20, seed=42):
    """
    Train H-AE at each lambda_phys value, compute metrics, generate plots.

    Args:
        latent_dim / lr : best values from hparam search
        Returns dict saved under "lambda_sweep" in final_metrics.json.
    """
    from train import train_model

    print("\n[Eval 6] Lambda sweep ...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir,      exist_ok=True)

    sweep_results = {}   # lambda_value → metrics dict
    models_dict   = {}   # lambda_value → trained HAE

    for lp in LAMBDA_SWEEP_VALUES:
        print(f"\n  Training H-AE with λ={lp} ...")
        import random; import numpy as np_
        random.seed(seed); np_.random.seed(seed); torch.manual_seed(seed)

        model = HAE(latent_dim=latent_dim).to(device)
        name  = f"hae_lambda{lp}"
        hist, _ = train_model(
            model, loaders, device,
            model_name     = name,
            checkpoint_dir = checkpoint_dir,
            log_path       = os.path.join(results_dir, "training_log.txt"),
            lambda_phys    = lp,
            lr             = lr,
            max_epochs     = max_epochs,
            patience       = patience,
            verbose        = True,
        )
        # Load best weights
        ckpt = os.path.join(checkpoint_dir, f"{name}_best.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))

        test_loader = loaders.get("test1") or loaders.get("test")
        _, y_te, xhat_te, _ = _collect_outputs(model, test_loader, device)
        mse     = _mse(xhat_te, y_te)
        overlap = _overlap(xhat_te, y_te)
        pc1     = compute_pc1_ratio(model, loaders["test"], device)

        sweep_results[str(lp)] = {
            "lambda_phys": lp,
            "test_mse":    mse,
            "overlap":     overlap,
            "pc1_ratio":   pc1,
        }
        models_dict[lp] = model
        print(f"  λ={lp}: test_mse={mse:.6f}  overlap={overlap:.4f}  pc1={pc1:.4f}")

    return sweep_results, models_dict


def plot_phase_space_lambda_sweep(models_dict, cae, test_loader, device,
                                  plots_dir, n_rows=5):
    """
    Grid: 4 columns (CAE, H-AE λ=0.01, 0.1, 1.0) × n_rows random test samples.
    Each cell: (q₀, p₀) trajectory coloured by time.
    Same axis limits within each row for fair visual comparison.
    Column titles include model name and PC1 ratio.
    """
    print("\n  Generating phase_space_lambda_sweep.png ...")

    # Collect latents
    _, _, _, z_cae = _collect_outputs(cae, test_loader, device)
    z_by_model = {lp: _collect_outputs(m, test_loader, device)[3]
                  for lp, m in models_dict.items()}

    N = z_cae.shape[0]
    torch.manual_seed(1)
    idx = torch.randperm(N)[:n_rows].numpy()

    lambdas = sorted(models_dict.keys())
    n_cols  = 1 + len(lambdas)          # CAE + one per lambda
    D       = next(iter(models_dict.values())).latent_dim

    # Pre-compute PC1 ratios for column titles
    pc1_cae = compute_pc1_ratio(cae, test_loader, device)
    pc1_by_lambda = {lp: compute_pc1_ratio(m, test_loader, device)
                     for lp, m in models_dict.items()}

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 4 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]

    col_labels = (["CAE"] +
                  [f"H-AE  λ={lp}" for lp in lambdas])
    pc1_labels = ([f"PC1={pc1_cae:.3f}"] +
                  [f"PC1={pc1_by_lambda[lp]:.3f}" for lp in lambdas])

    for col, (label, pc1_lbl) in enumerate(zip(col_labels, pc1_labels)):
        axes[0, col].set_title(f"{label}\n{pc1_lbl}",
                                fontsize=11, fontweight="bold", pad=8)

    for row, si in enumerate(idx):
        # Gather (a, b) pairs for this sample across all columns
        # CAE: channel 0 vs channel 1
        z_c = z_cae[si].numpy()
        ch1 = min(1, z_c.shape[0]-1)
        col_data = [(z_c[0], z_c[ch1])]
        for lp in lambdas:
            z_h = z_by_model[lp][si].numpy()
            col_data.append((z_h[0], z_h[D]))   # q₀ vs p₀

        # Compute per-row axis limits (union across all columns for fair comparison)
        all_a = np.concatenate([d[0] for d in col_data])
        all_b = np.concatenate([d[1] for d in col_data])
        pad_a = (all_a.max() - all_a.min()) * 0.1 + 1e-6
        pad_b = (all_b.max() - all_b.min()) * 0.1 + 1e-6
        xlim = (all_a.min() - pad_a, all_a.max() + pad_a)
        ylim = (all_b.min() - pad_b, all_b.max() + pad_b)

        for col, (a, b) in enumerate(col_data):
            ax = axes[row, col]
            T  = a.shape[0]
            sc = ax.scatter(a, b, c=np.arange(T), cmap="plasma",
                            s=10, rasterized=True)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", fontsize=9)
                ax.set_xlabel("z₀", fontsize=8)
                ax.set_ylabel("z₁", fontsize=8)
            else:
                ax.set_xlabel("q₀", fontsize=8)
                ax.set_ylabel("p₀", fontsize=8)
            ax.tick_params(labelsize=7)

    # Shared colorbar (time)
    cbar = fig.colorbar(sc, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.01)
    cbar.set_label("Latent time index", fontsize=9)

    fig.suptitle(
        "Phase-Space Trajectories: CAE vs H-AE at Different λ\n"
        "(colour = time along inspiral, same axis limits per row)",
        fontsize=13, fontweight="bold",
    )
    path = os.path.join(plots_dir, "phase_space_lambda_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_lambda_sweep_metrics(sweep_results, cae_metrics, plots_dir):
    """
    3-panel figure: test_mse / overlap / PC1 ratio vs lambda_phys (log x-axis).
    CAE shown as horizontal dashed reference line on each panel.
    Publication-quality formatting.
    """
    print("\n  Generating lambda_sweep_metrics.png ...")

    lambdas   = sorted(float(k) for k in sweep_results.keys())
    mses      = [sweep_results[str(lp)]["test_mse"]  for lp in lambdas]
    overlaps  = [sweep_results[str(lp)]["overlap"]   for lp in lambdas]
    pc1s      = [sweep_results[str(lp)]["pc1_ratio"] for lp in lambdas]

    # Support both legacy flat keys and new nested structure
    _test1_cae  = cae_metrics.get("reconstruction", {}).get("test1", {}).get("cae", {})
    cae_mse     = _test1_cae.get("mse")     or cae_metrics.get("test_mse_cae")
    cae_overlap = _test1_cae.get("overlap") or cae_metrics.get("overlap_cae")
    cae_pc1     = _test1_cae.get("pc1")     or cae_metrics.get("latent_pc1_ratio_cae")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.rcParams.update({"font.size": 11})

    panels = [
        (axes[0], mses,     cae_mse,     "Test MSE",          "MSE",           False),
        (axes[1], overlaps, cae_overlap, "Waveform Overlap",  "Overlap",       False),
        (axes[2], pc1s,     cae_pc1,     "Latent PC1 Ratio",  "PC1 ratio",     False),
    ]

    marker_kw = dict(marker="o", markersize=8, linewidth=2,
                     color="#2171B5", label="H-AE")
    ref_kw    = dict(color="#D94801", linewidth=1.5, linestyle="--",
                     label="CAE (reference)")

    for ax, vals, cae_val, title, ylabel, _ in panels:
        ax.semilogx(lambdas, vals, **marker_kw)
        if cae_val is not None:
            ax.axhline(cae_val, **ref_kw)
        ax.set_xlabel("λ (physics loss weight)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(lambdas)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle(
        "Effect of Physics Loss Weight λ on H-AE Performance\n"
        "(test set: both masses > 70 Msun)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(plots_dir, "lambda_sweep_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Physics variant loss curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_physics_variant_phase_space(models_dict, test_loader, device,
                                      plots_dir, cae=None, n_samples=5):
    """
    Latent phase-space trajectories (q₀ vs p₀) for ProperHAE and PortHAE variants.

    Saves two figures:
      physics_phase_space_proper.png  — [CAE ref] + ProperHAE at λ=0.1/0.5/1.0
      physics_phase_space_port.png    — [CAE ref] + PortHAE  at λ=0.1/0.5/1.0

    Args:
        models_dict : {"properhae_lambda0.1": model, "porthae_lambda0.5": model, ...}
                      Models must already have best-checkpoint weights loaded.
        cae         : optional reference BaselineCAE; shown as leftmost column.
    """
    print("\n[Physics variants] Plotting phase-space trajectories ...")
    os.makedirs(plots_dir, exist_ok=True)

    lambdas = [0.1, 0.5, 1.0]

    # Consistent sample indices across both figures
    torch.manual_seed(1)
    ref = next(iter(models_dict.values()), cae)
    if ref is None:
        print("  No models provided — skipping phase space.")
        return
    _, _, _, z_ref = _collect_outputs(ref, test_loader, device)
    N   = z_ref.shape[0]
    idx = torch.randperm(N)[:n_samples].numpy()

    # Pre-collect CAE latents once
    z_cae = None
    if cae is not None:
        _, _, _, z_cae = _collect_outputs(cae, test_loader, device)

    type_specs = [
        ("ProperHAE", "properhae", "physics_phase_space_proper.png"),
        ("PortHAE",   "porthae",   "physics_phase_space_port.png"),
    ]

    for type_label, key_prefix, filename in type_specs:
        # Build column specs: (header, z_all, is_physics_model, latent_dim)
        col_specs = []
        if cae is not None and z_cae is not None:
            pc1_c = compute_pc1_ratio(cae, test_loader, device)
            col_specs.append((f"CAE (ref)\nPC1={pc1_c:.3f}", z_cae, False, None))

        for lam in lambdas:
            key = f"{key_prefix}_lambda{lam}"
            m   = models_dict.get(key)
            if m is not None:
                _, _, _, z_m = _collect_outputs(m, test_loader, device)
                pc1 = compute_pc1_ratio(m, test_loader, device)
                col_specs.append((f"{type_label}  λ={lam}\nPC1={pc1:.3f}",
                                  z_m, True, m.latent_dim))
            else:
                col_specs.append((f"{type_label}  λ={lam}\n(missing)",
                                  None, True, 2))

        n_cols = len(col_specs)
        fig, axes = plt.subplots(n_samples, n_cols,
                                  figsize=(4.2 * n_cols, 4 * n_samples),
                                  constrained_layout=True)
        if n_samples == 1:
            axes = axes[None, :]

        for col, (label, *_) in enumerate(col_specs):
            axes[0, col].set_title(label, fontsize=10, fontweight="bold", pad=8)

        last_sc = None
        for row, si in enumerate(idx):
            # Gather (a, b) pairs for this sample
            pairs = []
            for (_, z_all, is_phys, D) in col_specs:
                if z_all is None:
                    pairs.append(None)
                elif is_phys:
                    zh = z_all[si].numpy()
                    pairs.append((zh[0], zh[D]))         # q₀ vs p₀
                else:
                    zc = z_all[si].numpy()
                    pairs.append((zc[0], zc[min(1, zc.shape[0]-1)]))  # z₀ vs z₁

            # Shared axis limits across all columns in this row
            valid = [p for p in pairs if p is not None]
            if valid:
                all_a = np.concatenate([p[0] for p in valid])
                all_b = np.concatenate([p[1] for p in valid])
                pad_a = (all_a.max() - all_a.min()) * 0.1 + 1e-6
                pad_b = (all_b.max() - all_b.min()) * 0.1 + 1e-6
                xlim = (all_a.min() - pad_a, all_a.max() + pad_a)
                ylim = (all_b.min() - pad_b, all_b.max() + pad_b)
            else:
                xlim = ylim = (-1, 1)

            for col, (pair, (_, _, is_phys, _D)) in enumerate(zip(pairs, col_specs)):
                ax = axes[row, col]
                if pair is None:
                    ax.text(0.5, 0.5, "no model", ha="center", va="center",
                            transform=ax.transAxes, color="grey")
                    continue
                a, b = pair
                last_sc = ax.scatter(a, b, c=np.arange(a.shape[0]),
                                     cmap="plasma", s=10, rasterized=True)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_xlabel("z₀" if not is_phys else "q₀", fontsize=8)
                ax.set_ylabel("z₁" if not is_phys else "p₀", fontsize=8)
                ax.tick_params(labelsize=7)

        if last_sc is not None:
            cbar = fig.colorbar(last_sc, ax=axes,
                                orientation="vertical", fraction=0.015, pad=0.01)
            cbar.set_label("Latent time index", fontsize=9)

        fig.suptitle(
            f"Latent Phase-Space Trajectories: CAE vs {type_label} variants\n"
            "(colour = time along inspiral, same axis limits per row)",
            fontsize=13, fontweight="bold",
        )
        path = os.path.join(plots_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_physics_variant_curves(histories: dict, plots_dir: str):
    """
    Loss curves for ProperHAE and PortHAE variants across lambda values.

    Args:
        histories : {model_name: {"train_recon": [...], "val_recon": [...],
                                  "train_phys":  [...], "val_phys":  [...]}}
                    Expected keys follow run_physics_variants naming:
                    "properhae_lambda{lam}" and "porthae_lambda{lam}".

    Saves: plots_dir/physics_variant_loss_curves.png
           2-row × 3-col grid (ProperHAE / PortHAE) × (λ=0.1 / 0.5 / 1.0).
           Each subplot: train_recon, val_recon, train_phys, val_phys on log scale.
    """
    print("\n[Physics variants] Plotting loss curves ...")
    os.makedirs(plots_dir, exist_ok=True)

    lambdas    = [0.1, 0.5, 1.0]
    row_specs  = [("ProperHAE", "properhae"), ("PortHAE", "porthae")]
    n_rows, n_cols = len(row_specs), len(lambdas)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             constrained_layout=True)

    for r, (model_label, key_prefix) in enumerate(row_specs):
        for c, lam in enumerate(lambdas):
            ax  = axes[r, c]
            key = f"{key_prefix}_lambda{lam}"
            h   = histories.get(key)

            if h is None or not h.get("train_recon"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_title(f"{model_label}  λ={lam}", fontsize=10)
                continue

            ep = np.arange(1, len(h["train_recon"]) + 1)

            ax.semilogy(ep, h["train_recon"], color="#2171B5",
                        lw=1.5, label="train recon")
            ax.semilogy(ep, h["val_recon"],   color="#2171B5",
                        lw=1.5, linestyle="--", label="val recon")

            phys_tr = h.get("train_phys", [])
            phys_vl = h.get("val_phys",   [])
            if phys_tr and any(v > 0 for v in phys_tr):
                ax.semilogy(ep, phys_tr, color="#D94801",
                            lw=1.2, alpha=0.8, label="train physics")
                ax.semilogy(ep, phys_vl, color="#D94801",
                            lw=1.2, linestyle="--", alpha=0.8, label="val physics")

            ax.set_title(f"{model_label}  λ={lam}  "
                         f"(best val={min(h['val_recon']):.2e}  ep={len(ep)})",
                         fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Loss (log scale)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, which="both", alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "ProperHAE & PortHAE — Training Loss Curves\n"
        "(solid = train, dashed = val  |  blue = reconstruction, orange = physics)",
        fontsize=12, fontweight="bold",
    )
    path = os.path.join(plots_dir, "physics_variant_loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation entry point
# ──────────────────────────────────────────────────────────────────────────────

def _compute_model_metrics(model, loader, device):
    """MSE, overlap, PC1 for one model on one loader."""
    _, y, xhat, _ = _collect_outputs(model, loader, device)
    return {
        "mse":     _mse(xhat, y),
        "overlap": _overlap(xhat, y),
        "pc1":     compute_pc1_ratio(model, loader, device),
    }


def run_evaluation(hae, cae, loaders, info, device,
                   hae_history, cae_history,
                   physics_models=None,
                   plots_dir="./plots", results_dir="./results"):
    """
    Run experiments 1–5 and save plots + metrics JSON.
    Experiment 6 (lambda sweep) is triggered separately from main.py.

    Args:
        physics_models : optional dict {"properhae": model, "porthae": model}
                         These are included in the metrics table but skip
                         reconstruction plots (latent plots handled separately).
    """
    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Primary test loader for plots (test1 preferred, fall back to legacy "test")
    test1_loader = loaders.get("test1") or loaders.get("test")
    test2_loader = loaders.get("test2")
    test1_idx    = info.get("test1_idx", info.get("test_idx"))
    test2_idx    = info.get("test2_idx")

    # 1. Reconstruction plots — HAE and CAE on test1 only
    recon_metrics = eval_reconstruction(
        hae, cae, test1_loader, device, plots_dir,
        train_eval_loader=loaders.get("train_eval"),
    )

    # 2. Phase space — HAE and CAE on test1
    eval_phase_space(hae, cae, test1_loader, device, plots_dir)

    # 3. Generalization bar chart
    gen_metrics = eval_generalization(recon_metrics, plots_dir)

    # 4. Loss curves
    plot_loss_curves(hae_history, cae_history, plots_dir)

    # 5. Noise robustness — HAE and CAE on each available test set
    noise_metrics = {}
    for ts_name, ts_loader, ts_idx in [
        ("test1", test1_loader, test1_idx),
        ("test2", test2_loader, test2_idx),
    ]:
        if ts_loader is None or ts_idx is None:
            continue
        print(f"\n[Eval 5] Noise robustness ({ts_name}) ...")
        nm = eval_noise_robustness(
            hae, cae,
            test_idx    = ts_idx,
            hdf_path    = info["hdf_path"],
            test_loader = ts_loader,
            device      = device,
            plots_dir   = plots_dir,
            plot_suffix = f"_{ts_name}",
        )
        noise_metrics[ts_name] = nm

    # ── Comprehensive metrics: all models × all test sets ──────────────────
    all_models = {"hae": hae, "cae": cae}
    if physics_models:
        all_models.update(physics_models)

    test_sets = {k: v for k, v in [("test1", test1_loader), ("test2", test2_loader)]
                 if v is not None}

    reconstruction_metrics = {}
    print(f"\n{'='*65}")
    print(f"  {'Model':<14}  {'Test set':<8}  {'MSE':>10}  {'Overlap':>8}  {'PC1':>6}")
    print(f"{'='*65}")
    for ts_name, ts_loader in test_sets.items():
        reconstruction_metrics[ts_name] = {}
        for model_name, model in all_models.items():
            m = _compute_model_metrics(model, ts_loader, device)
            reconstruction_metrics[ts_name][model_name] = m
            print(f"  {model_name:<14}  {ts_name:<8}  "
                  f"{m['mse']:>10.6f}  {m['overlap']:>8.4f}  {m['pc1']:>6.4f}")
    print(f"{'='*65}")

    # Train MSE (HAE and CAE)
    train_metrics = {}
    if loaders.get("train_eval"):
        _, y_tr, xhat_tr_h, _ = _collect_outputs(hae, loaders["train_eval"], device)
        _, _,    xhat_tr_c, _ = _collect_outputs(cae, loaders["train_eval"], device)
        train_metrics = {
            "hae": {"mse": _mse(xhat_tr_h, y_tr)},
            "cae": {"mse": _mse(xhat_tr_c, y_tr)},
        }

    all_metrics = {
        "reconstruction":  reconstruction_metrics,
        "train":           train_metrics,
        "noise_robustness": noise_metrics,
    }

    json_path = os.path.join(results_dir, "final_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {json_path}")

    return all_metrics
