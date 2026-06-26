"""
regenerate_phase_space_2x2.py
Produces ./plots/phase_space_cae_vs_hae.png:
  2 rows (samples) x 2 columns (CAE | H-AE lambda=0.1)
Identical style to phase_space_lambda_sweep.png — just fewer panels.
Does NOT retrain or modify any other file.

Usage:
    python src/regenerate_phase_space_2x2.py
"""

import os, sys, json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data   import make_splits
from models import HAE, BaselineCAE

# ── seeds ─────────────────────────────────────────────────────────────────────
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── rcParams (same as regenerate_plots.py) ────────────────────────────────────
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

CKPT_DIR    = "./checkpoints"
PLOTS_DIR   = "./plots"
METRICS_PATH = "./results/final_metrics.json"
OUT_PATH    = os.path.join(PLOTS_DIR, "phase_space_cae_vs_hae.png")
LATENT_DIM  = 2
N_ROWS      = 2   # samples
N_COLS      = 2   # CAE | H-AE λ=0.1

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── load PC1 ratios from saved metrics ───────────────────────────────────────
with open(METRICS_PATH) as f:
    metrics = json.load(f)
cae_pc1 = metrics["latent_pc1_ratio_cae"]
hae_pc1 = metrics["lambda_sweep"]["0.1"]["pc1_ratio"]

# ── data ──────────────────────────────────────────────────────────────────────
print("Loading data ...")
loaders, info = make_splits("./output/dataset.hdf", batch_size=32)
device = torch.device("cpu")

# ── column specs: (title, ModelClass, ckpt filename, pc1) ────────────────────
col_specs = [
    ("CAE",                BaselineCAE, "cae_best.pt",            cae_pc1),
    (r"H-AE $\lambda$=0.1", HAE,       "hae_lambda0.1_best.pt",  hae_pc1),
]

# ── fixed sample indices (same RNG state as the 4-column figure) ─────────────
torch.manual_seed(SEED)
row_idx = torch.randperm(info["n_test"])[:N_ROWS].tolist()

# ── inference helper ──────────────────────────────────────────────────────────
@torch.no_grad()
def collect_latents(model, loader):
    z_list = []
    model.eval()
    for x, _ in loader:
        if isinstance(model, HAE):
            _, q, p, _ = model(x)
            z = torch.cat([q, p], dim=1)
        else:
            _, z = model(x)
        for i in range(x.shape[0]):
            z_list.append(z[i].cpu().numpy())
    return z_list

# ── collect latents for both models ──────────────────────────────────────────
all_latents = {}
for (label, ModelClass, ckpt_name, _) in col_specs:
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: {ckpt_path} not found — skipping.")
        all_latents[label] = None
        continue
    model = ModelClass(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    all_latents[label] = collect_latents(model, loaders["test"])
    print(f"  Loaded {ckpt_name}")

# ── plot ──────────────────────────────────────────────────────────────────────
cmap = plt.get_cmap("viridis")

fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(3.6, 3.6),
                          gridspec_kw={"hspace": 0.12, "wspace": 0.08})

# Column titles
for col_i, (label, ModelClass, _, pc1_val) in enumerate(col_specs):
    axes[0, col_i].set_title(f"{label}\nPC1={pc1_val:.3f}", fontsize=9, pad=4)

# Draw trajectories
for col_i, (label, ModelClass, _, _pc1) in enumerate(col_specs):
    z_all = all_latents.get(label)
    for row_i, si in enumerate(row_idx):
        ax = axes[row_i, col_i]

        if z_all is None or si >= len(z_all):
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
        else:
            z  = z_all[si]
            q0 = z[0]
            p0 = z[LATENT_DIM] if ModelClass == HAE else (z[1] if z.shape[0] > 1 else z[0])
            T  = q0.shape[0]

            for t in range(T - 1):
                ax.plot(q0[t:t+2], p0[t:t+2],
                        color=cmap(t / T), linewidth=0.8, alpha=0.85)
            ax.plot(q0[0],  p0[0],  "o", color=cmap(0.0), markersize=4, zorder=5)
            ax.plot(q0[-1], p0[-1], "s", color=cmap(1.0), markersize=4, zorder=5)

        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        if col_i == 0:
            ax.set_ylabel(f"sample {row_i+1}", fontsize=9, labelpad=3)

# Shared axis limits per row
for row_i in range(N_ROWS):
    si = row_idx[row_i]
    all_q, all_p = [], []
    for col_i, (label, ModelClass, _, _) in enumerate(col_specs):
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
    for col_i in range(N_COLS):
        axes[row_i, col_i].set_xlim(q_cat.min() - pad_q, q_cat.max() + pad_q)
        axes[row_i, col_i].set_ylim(p_cat.min() - pad_p, p_cat.max() + pad_p)

# x-axis label on bottom row centre
axes[-1, 0].set_xlabel(r"$q_0$", fontsize=9)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.04, pad=0.02, shrink=0.85)
cbar.set_label("Latent time (norm.)", fontsize=9)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(["early", "mid", "late"])

fig.savefig(OUT_PATH)
plt.close(fig)
print(f"saved {OUT_PATH}")
