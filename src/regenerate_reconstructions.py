"""
regenerate_reconstructions.py
Produces ./plots/reconstructions.png — 3-row overlay figure for a two-column
LaTeX report.  Loads checkpoints, does NOT retrain anything.

Usage:
    python src/regenerate_reconstructions.py
"""

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data   import make_splits
from models import HAE, BaselineCAE

# ── seeds ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── rcParams ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "serif",
})

# ── paths ─────────────────────────────────────────────────────────────────────
CKPT_HAE = "./checkpoints/hae_lambda0.1_best.pt"
CKPT_CAE = "./checkpoints/cae_best.pt"
OUT_PATH = "./plots/reconstructions.png"
os.makedirs("./plots", exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
loaders, _ = make_splits("./output/dataset.hdf", batch_size=32)

# ── models ────────────────────────────────────────────────────────────────────
hae = HAE(latent_dim=2)
cae = BaselineCAE(latent_dim=2)
hae.load_state_dict(torch.load(CKPT_HAE, map_location="cpu"))
cae.load_state_dict(torch.load(CKPT_CAE, map_location="cpu"))
hae.eval(); cae.eval()

# ── inference on first 3 test samples ────────────────────────────────────────
with torch.no_grad():
    for x, y in loaders["test"]:
        x3, y3 = x[:3], y[:3]
        xhat_hae = hae(x3)[0]          # HAE returns (x_hat, q, p, H_t)
        xhat_cae = cae(x3)[0]          # CAE returns (x_hat, z)
        break

# Convert to numpy, drop channel dim
y3       = y3[:, 0, :].numpy()
xhat_hae = xhat_hae[:, 0, :].detach().numpy()
xhat_cae = xhat_cae[:, 0, :].detach().numpy()
x_idx    = np.arange(y3.shape[1])

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.2), sharex=True)

for i, ax in enumerate(axes):
    # Target plotted first (behind), reconstructions on top
    ax.plot(x_idx, xhat_cae[i], color="tab:orange", lw=1.0,
            linestyle="--", zorder=2, label="CAE")
    ax.plot(x_idx, xhat_hae[i], color="tab:blue",   lw=1.0,
            zorder=3, label="H-AE")
    ax.plot(x_idx, y3[i],       color="black",       lw=1.2,
            zorder=1, alpha=0.55, label="target")

    ax.set_ylabel(f"sample {i+1}", fontsize=9)

    if i == 0:
        ax.legend(loc="upper right", frameon=False,
                  handlelength=1.4, handletextpad=0.4,
                  borderpad=0, labelspacing=0.25)

axes[-1].set_xlabel("sample index")
plt.tight_layout()
fig.savefig(OUT_PATH)
plt.close(fig)
print(f"saved {OUT_PATH}")
