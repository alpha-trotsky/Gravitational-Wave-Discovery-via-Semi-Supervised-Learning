"""
evaluation/config.py — Paths and model registry for all evaluation scripts.

Each eval script imports this and calls load_all_models() to get whatever
checkpoints are present, skipping any that are missing.
"""

import os
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
HDF_PATH       = "./output/dataset.hdf"
CHECKPOINT_DIR = "./checkpoints"
PLOTS_DIR      = "./plots"
RESULTS_DIR    = "./results"
BEST_CFG_PATH  = "./results/best_configs.json"

BATCH_SIZE = 16

# Must match training/config.py LAMBDA_VALUES
LAMBDA_VALUES = [0.1, 0.25, 0.5, 1.0]


# ── Checkpoint path helpers ───────────────────────────────────────────────────

def ckpt_path(model_family: str, lam=None, checkpoint_dir=CHECKPOINT_DIR):
    """
    Returns the checkpoint path for a given model family and lambda.

    Examples:
      ckpt_path("hae", 0.1)   → checkpoints/hae/lambda0.1_best.pt
      ckpt_path("cae")        → checkpoints/cae/best.pt
    """
    if model_family == "cae":
        return os.path.join(checkpoint_dir, "cae", "best.pt")
    return os.path.join(checkpoint_dir, model_family, f"lambda{lam}_best.pt")


# ── Model registry ────────────────────────────────────────────────────────────

def build_model_registry(checkpoint_dir=CHECKPOINT_DIR,
                         results_dir=RESULTS_DIR,
                         lambda_values=LAMBDA_VALUES):
    """
    Return a list of (display_name, model_class, latent_dim, checkpoint_path)
    for every model × lambda combination.

    Latent dims are read from best_configs.json; fall back to 2.
    """
    from core.models import HAE, BaselineCAE, ProperHAE, PortHAE

    cfg_path = os.path.join(results_dir, "best_configs.json")
    cfgs = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfgs = json.load(f)

    def _ld(key):
        return int(cfgs.get(key, {}).get("latent_dim", 2))

    registry = []

    # HAE at each lambda
    for lam in lambda_values:
        registry.append((
            f"HAE λ={lam}", HAE, _ld("hae"),
            ckpt_path("hae", lam, checkpoint_dir),
        ))

    # CAE — single run
    registry.append((
        "CAE", BaselineCAE, _ld("cae"),
        ckpt_path("cae", checkpoint_dir=checkpoint_dir),
    ))

    # ProperHAE at each lambda
    for lam in lambda_values:
        registry.append((
            f"ProperHAE λ={lam}", ProperHAE, _ld("properhae"),
            ckpt_path("properhae", lam, checkpoint_dir),
        ))

    # PortHAE at each lambda
    for lam in lambda_values:
        registry.append((
            f"PortHAE λ={lam}", PortHAE, _ld("porthae"),
            ckpt_path("porthae", lam, checkpoint_dir),
        ))

    return registry
