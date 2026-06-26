"""
regenerate_plots.py -- Produce three publication-quality figures for the report.

Loads existing checkpoints and final_metrics.json — does NOT retrain anything.

Usage:
    python src/regenerate_plots.py

Outputs (overwrite existing):
    ./plots/lambda_sweep_metrics.png
    ./plots/phase_space_lambda_sweep.png
    ./plots/reconstructions.png
"""

import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data   import make_splits, CROP_LEN, SAMPLING_RATE
from models import HAE, BaselineCAE

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
HDF_PATH      = "./output/dataset.hdf"
CKPT_DIR      = "./checkpoints"
PLOTS_DIR     = "./plots"
METRICS_PATH  = "./results/final_metrics.json"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "lines.linewidth":    1.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "serif",
})

# ── colours ───────────────────────────────────────────────────────────────────
C_HAE = "#2171B5"   # blue for H-AE
C_CAE = "#D94801"   # orange-red for CAE


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model, ckpt_path, device):
    """Load state dict; return False and print warning if file missing."""
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: checkpoint not found: {ckpt_path}  -- skipping.")
        return False
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return True


@torch.no_grad()
def get_latents_and_recon(model, loader, device, n_samples=None):
    """
    Run model over loader.  Returns:
        z_list   : list of (2*D, T) numpy arrays, one per sample
        xhat_list: list of (1792,) numpy arrays
        y_list   : list of (1792,) numpy arrays
    Stops after n_samples if given.
    """
    z_list, xhat_list, y_list = [], [], []
    for x, y in loader:
        x = x.to(device)
        if isinstance(model, HAE):
            xhat, q, p, _ = model(x)
            z = torch.cat([q, p], dim=1)   # (B, 2D, T)
        else:
            xhat, z = model(x)
        for i in range(x.shape[0]):
            z_list.append(z[i].cpu().numpy())
            xhat_list.append(xhat[i, 0].cpu().numpy())
            y_list.append(y[i, 0].numpy())
            if n_samples is not None and len(z_list) >= n_samples:
                return z_list, xhat_list, y_list
    return z_list, xhat_list, y_list


# ══════════════════════════════════════════════════════════════════════════════
# Load data once
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data ...")
loaders, info = make_splits(HDF_PATH, batch_size=32)
device = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════════════════
# Load metrics
# ══════════════════════════════════════════════════════════════════════════════
with open(METRICS_PATH) as f:
    metrics = json.load(f)

sweep    = metrics["lambda_sweep"]          # {"0.01": {...}, "0.1": {...}, "1.0": {...}}
lambdas  = [0.01, 0.1, 1.0]
cae_mse  = metrics["test_mse_cae"]
cae_olap = metrics["overlap_cae"]
cae_pc1  = metrics["latent_pc1_ratio_cae"]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 -- lambda_sweep_metrics.png
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 1: lambda_sweep_metrics.png ...")

mses     = [sweep[str(lp)]["test_mse"]   for lp in lambdas]
overlaps = [sweep[str(lp)]["overlap"]    for lp in lambdas]
pc1s     = [sweep[str(lp)]["pc1_ratio"]  for lp in lambdas]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

panel_data = [
    (axes[0], mses,     cae_mse,  "Test MSE",        True,  "MSE"),
    (axes[1], overlaps, cae_olap, "Waveform overlap", False, "Overlap"),
    (axes[2], pc1s,     cae_pc1,  "Latent PC1 ratio", False, "PC1 ratio"),
]

hae_handle = cae_handle = None
for ax, vals, cae_val, title, log_y, ylabel in panel_data:
    # H-AE line
    h1, = ax.plot(lambdas, vals, color=C_HAE, marker="o",
                  markersize=6, linestyle="-", label="H-AE", zorder=3)
    # Highlight lambda=0.1
    best_idx = lambdas.index(0.1)
    ax.plot(lambdas[best_idx], vals[best_idx], color=C_HAE,
            marker="o", markersize=10, markerfacecolor="white",
            markeredgewidth=2.0, zorder=4)
    # CAE baseline
    x_range = [min(lambdas) / 1.5, max(lambdas) * 1.5]
    h2, = ax.plot(x_range, [cae_val, cae_val], color=C_CAE,
                  linestyle="--", linewidth=1.2, label="CAE baseline", zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(lambdas)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlim(0.006, 1.8)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_\mathrm{phys}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)

    hae_handle, cae_handle = h1, h2

# Shared legend at top of figure
fig.legend(handles=[hae_handle, cae_handle],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06),
           frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(PLOTS_DIR, "lambda_sweep_metrics.png")
fig.savefig(out1)
plt.close(fig)
print(f"  Saved: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 -- phase_space_lambda_sweep.png
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 2: phase_space_lambda_sweep.png ...")

LATENT_DIM = 2   # from best_configs.json

# Checkpoint map: column label -> (model_class, ckpt_path, pc1_from_metrics)
col_specs = [
    ("CAE",              BaselineCAE, "cae_best.pt",         cae_pc1),
    (r"H-AE $\lambda$=0.01", HAE,    "hae_lambda0.01_best.pt", sweep["0.01"]["pc1_ratio"]),
    (r"H-AE $\lambda$=0.1",  HAE,    "hae_lambda0.1_best.pt",  sweep["0.1"]["pc1_ratio"]),
    (r"H-AE $\lambda$=1.0",  HAE,    "hae_lambda1.0_best.pt",  sweep["1.0"]["pc1_ratio"]),
]

N_ROWS = 3

# Choose 3 fixed test-sample indices
torch.manual_seed(SEED)
n_test  = info["n_test"]
row_idx = torch.randperm(n_test)[:N_ROWS].tolist()

# Collect latents per model
all_latents = {}   # col_label -> list of (2D, T) arrays for all test samples
for (label, ModelClass, ckpt_name, _pc1) in col_specs:
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    model = ModelClass(latent_dim=LATENT_DIM).to(device)
    if not load_model(model, ckpt_path, device):
        all_latents[label] = None
        continue
    z_list, _, _ = get_latents_and_recon(model, loaders["test"], device)
    all_latents[label] = z_list

fig, axes = plt.subplots(N_ROWS, 4,
                          figsize=(7.0, 5.0),
                          gridspec_kw={"hspace": 0.12, "wspace": 0.08})

cmap = plt.get_cmap("viridis")

for col_i, (label, ModelClass, _ckpt, pc1_val) in enumerate(col_specs):
    # Column title (top row only)
    axes[0, col_i].set_title(
        f"{label}\nPC1={pc1_val:.3f}",
        fontsize=9, pad=4,
    )

    z_all = all_latents.get(label)

    for row_i, si in enumerate(row_idx):
        ax = axes[row_i, col_i]

        if z_all is None or si >= len(z_all):
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
        else:
            z = z_all[si]          # (2D, T)
            # First canonical pair: q[0] vs p[0]
            if isinstance(ModelClass(), HAE) if False else (ModelClass == HAE):
                q0 = z[0]              # first q
                p0 = z[LATENT_DIM]     # corresponding p
            else:
                q0 = z[0]
                p0 = z[1] if z.shape[0] > 1 else z[0]

            T      = q0.shape[0]
            colors = cmap(np.linspace(0, 1, T))

            # Thin line coloured by time
            for t in range(T - 1):
                ax.plot(q0[t:t+2], p0[t:t+2],
                        color=cmap(t / T), linewidth=0.8, alpha=0.85)

            # Start marker
            ax.plot(q0[0], p0[0], "o", color=cmap(0.0),
                    markersize=4, zorder=5)
            # End marker
            ax.plot(q0[-1], p0[-1], "s", color=cmap(1.0),
                    markersize=4, zorder=5)

        # Clean grid look: remove tick labels
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        # Row label on leftmost column
        if col_i == 0:
            ax.set_ylabel(f"sample {row_i+1}", fontsize=9, labelpad=3)

# Share axis limits within each row for fair comparison
for row_i in range(N_ROWS):
    si = row_idx[row_i]
    # Collect all q0, p0 across columns for this row
    all_q, all_p = [], []
    for col_i, (label, ModelClass, _ckpt, _pc1) in enumerate(col_specs):
        z_all = all_latents.get(label)
        if z_all is None or si >= len(z_all):
            continue
        z = z_all[si]
        all_q.append(z[0])
        all_p.append(z[LATENT_DIM] if ModelClass == HAE else (z[1] if z.shape[0] > 1 else z[0]))
    if not all_q:
        continue
    q_cat = np.concatenate(all_q); p_cat = np.concatenate(all_p)
    pad_q = (q_cat.max() - q_cat.min()) * 0.12 + 1e-6
    pad_p = (p_cat.max() - p_cat.min()) * 0.12 + 1e-6
    xlim = (q_cat.min() - pad_q, q_cat.max() + pad_q)
    ylim = (p_cat.min() - pad_p, p_cat.max() + pad_p)
    for col_i in range(4):
        axes[row_i, col_i].set_xlim(xlim)
        axes[row_i, col_i].set_ylim(ylim)

# Axis label on bottom row, centre column
axes[-1, 1].set_xlabel(r"$q_0$  (latent position)", fontsize=9)

# Colorbar for time
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.02, pad=0.01, shrink=0.85)
cbar.set_label("Latent time (norm.)", fontsize=9)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(["early", "mid", "late"])

out2 = os.path.join(PLOTS_DIR, "phase_space_lambda_sweep.png")
fig.savefig(out2)
plt.close(fig)
print(f"  Saved: {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 -- reconstructions.png
# Dual-panel per sample: waveform overlay (top) + residuals (bottom).
# Layout: 3 samples × 2 subpanels = 6 rows via GridSpec height_ratios [3,1,...].
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 3: reconstructions.png ...")

hae_best = HAE(latent_dim=LATENT_DIM).to(device)
cae_best = BaselineCAE(latent_dim=LATENT_DIM).to(device)

ok_hae = load_model(hae_best, os.path.join(CKPT_DIR, "hae_lambda0.1_best.pt"), device)
ok_cae = load_model(cae_best, os.path.join(CKPT_DIR, "cae_best.pt"),            device)

if not (ok_hae and ok_cae):
    print("  Skipping Figure 3 -- missing checkpoints.")
else:
    _, xhat_hae, y_clean = get_latents_and_recon(hae_best, loaders["test"], device)
    _, xhat_cae, _       = get_latents_and_recon(cae_best, loaders["test"], device)

    # Use the same fixed samples as Figure 2
    sample_idx = row_idx[:3]

    # Time axis: eval crop = [CROP_START, CROP_END] = [EVENT_IDX-2048, EVENT_IDX-256]
    # Express as seconds before merger (negative throughout)
    t_start = -(2048 / SAMPLING_RATE)   # -1.000 s
    t_end   = -(256  / SAMPLING_RATE)   # -0.125 s
    t_axis  = np.linspace(t_start, t_end, CROP_LEN)

    N_SAMPLES = 3
    # GridSpec: pairs of (main, residual) rows, height ratio 3:1
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(3.4, 5.2))
    # 6 rows: [main, resid, main, resid, main, resid]
    hr = [3, 1] * N_SAMPLES
    gs = GridSpec(2 * N_SAMPLES, 1, figure=fig,
                  height_ratios=hr, hspace=0.0)

    for row_i, si in enumerate(sample_idx):
        ax_main = fig.add_subplot(gs[2 * row_i])
        ax_res  = fig.add_subplot(gs[2 * row_i + 1], sharex=ax_main)

        y  = y_clean[si]
        yh = xhat_hae[si]
        yc = xhat_cae[si]

        # ── Main panel: target (gray bg) + H-AE + CAE on top ─────────────
        ax_main.plot(t_axis, y,  color="0.65", lw=1.4, zorder=1,
                     label="Target")
        ax_main.plot(t_axis, yh, color=C_HAE,  lw=1.0, zorder=3,
                     label=r"H-AE ($\lambda$=0.1)")
        ax_main.plot(t_axis, yc, color=C_CAE,  lw=1.0, linestyle="--",
                     zorder=2, label="CAE")

        ax_main.set_ylabel("strain", fontsize=8, labelpad=2)
        ax_main.tick_params(labelbottom=False, labelsize=8)
        ax_main.tick_params(axis="y", labelsize=7)

        # Legend in top panel only
        if row_i == 0:
            ax_main.legend(loc="upper left", frameon=False,
                           handlelength=1.4, handletextpad=0.3,
                           borderpad=0, labelspacing=0.25)

        # Sample label inside panel
        ax_main.text(0.98, 0.92, f"sample {row_i+1}",
                     transform=ax_main.transAxes,
                     ha="right", va="top", fontsize=8, color="0.4")

        # ── Residual panel ────────────────────────────────────────────────
        ax_res.axhline(0, color="0.5", lw=0.6, zorder=1)
        ax_res.plot(t_axis, yh - y, color=C_HAE, lw=0.8, zorder=3)
        ax_res.plot(t_axis, yc - y, color=C_CAE, lw=0.8, linestyle="--", zorder=2)
        ax_res.set_ylabel("res.", fontsize=7, labelpad=2)
        ax_res.tick_params(axis="y", labelsize=6)

        # Only show x ticks on the last residual panel
        if row_i < N_SAMPLES - 1:
            ax_res.tick_params(labelbottom=False)
        else:
            ax_res.set_xlabel("Time before merger (s)", fontsize=9)
            ax_res.tick_params(axis="x", labelsize=8)

        # Remove gap between main and residual
        plt.setp(ax_main.get_xticklabels(), visible=False)

    out3 = os.path.join(PLOTS_DIR, "reconstructions.png")
    fig.savefig(out3)
    plt.close(fig)
    print(f"  Saved: {out3}")

print("\nAll figures generated.")
