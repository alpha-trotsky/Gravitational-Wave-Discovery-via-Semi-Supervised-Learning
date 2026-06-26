"""
config.py — Shared constants and lightweight utilities used across the pipeline.

Centralising these here avoids circular imports between main.py and helper
scripts (run_physics_variants.py, evaluate_ood.py, etc.) that previously
imported from main.py.
"""

import random
import warnings

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
HDF_PATH       = "./output/dataset.hdf"
CHECKPOINT_DIR = "./checkpoints"
PLOTS_DIR      = "./plots"
RESULTS_DIR    = "./results"
LOG_PATH       = "./results/training_log.txt"
BEST_CFG_PATH  = "./results/best_configs.json"

# ── Training defaults ─────────────────────────────────────────────────────────
SEED        = 42
BATCH_SIZE  = 16
MAX_EPOCHS  = 500
PATIENCE    = 200

# ── Model defaults (used when best_configs.json is absent) ────────────────────
DEFAULT_HAE_CFG       = {"lambda_phys": 0.1, "latent_dim": 2, "lr": 1e-3}
DEFAULT_CAE_CFG       = {"latent_dim": 2, "lr": 1e-3}
DEFAULT_PROPERHAE_CFG = {"lambda_phys": 0.5, "latent_dim": 2, "lr": 1e-3}
DEFAULT_PORTHAE_CFG   = {"lambda_phys": 1.0, "latent_dim": 2, "lr": 1e-3}


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        warnings.warn("torch.use_deterministic_algorithms(True) not fully supported "
                      "on this platform — some ops may be non-deterministic.")


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("CUDA not available — using CPU")
    return dev
