"""
training/config.py — All hyperparameters and paths for the training pipeline.

Edit this file to change lambda values, search grids, epoch counts, or paths
before running hparam searches or train_all.py on Kaggle.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
HDF_PATH       = "./output/dataset.hdf"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"
LOG_PATH       = "./results/training_log.txt"
BEST_CFG_PATH  = "./results/best_configs.json"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED       = 42
BATCH_SIZE = 16

# ── Full training run ─────────────────────────────────────────────────────────
MAX_EPOCHS = 500
PATIENCE   = 200

# ── Lambda values for the full sweep (HAE, ProperHAE, PortHAE) ───────────────
LAMBDA_VALUES = [0.1, 0.25, 0.5, 1.0]

# ── Hparam search — shared ────────────────────────────────────────────────────
HPARAM_EPOCHS   = 50
HPARAM_PATIENCE = 10
# Lambda is fixed during hparam search so we only search lr and latent_dim.
# The sweep over lambda happens at full training time (LAMBDA_VALUES above).
HPARAM_LAMBDA_FIXED = 0.1

LATENT_DIM_GRID = [1, 2, 4]

# ── Hparam search — HAE grid (applied to CAE too) ────────────────────────────
# CAE shares the same backbone so we use HAE's best latent_dim and lr.
HAE_LR_GRID = [1e-4, 1e-3]

# ── Hparam search — physics models grid (ProperHAE → applied to PortHAE) ─────
# Second-order autograd through the Hamiltonian MLP produces larger/noisier
# gradients, so the lr search skews lower than the variance-based HAE.
PHYSICS_LR_GRID = [5e-5, 1e-4, 5e-4]

# ── Default configs (used when best_configs.json is absent) ──────────────────
DEFAULT_HAE_CFG     = {"latent_dim": 2, "lr": 1e-3}
DEFAULT_PHYSICS_CFG = {"latent_dim": 2, "lr": 1e-4}
