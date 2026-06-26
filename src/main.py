"""
main.py — End-to-end pipeline entry point.

Scientific story:
  Train H-AE (Hamiltonian Autoencoder) and baseline CAE purely on clean BBH
  inspiral signals (h1_signal). The Hamiltonian prior encourages the latent space
  to respect approximate energy conservation in the adiabatic pre-merger regime.
  Then evaluate whether the learned clean-signal manifold generalises to:
    (a) held-out mass ranges (mass > 50 M☉) — in-distribution OOD
    (b) whitened noisy detector strain (h1_strain) — honest OOD evaluation

Usage:
  python src/main.py --run-all           # full pipeline (recommended)
  python src/main.py --hparam-search     # grid search only
  python src/main.py --train             # final training only
  python src/main.py --evaluate          # evaluation only (loads checkpoints)
  python src/main.py --run-all --no-hparam  # skip grid search, use defaults
"""

import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    get_device, set_seeds,
    HDF_PATH, CHECKPOINT_DIR, PLOTS_DIR, RESULTS_DIR, LOG_PATH, BEST_CFG_PATH,
    SEED, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
    DEFAULT_HAE_CFG, DEFAULT_CAE_CFG, DEFAULT_PROPERHAE_CFG, DEFAULT_PORTHAE_CFG,
)
from data          import make_splits
from models        import HAE, BaselineCAE, ProperHAE, PortHAE
from train         import train_model
from hparam_search import run_hparam_search
from evaluate      import (run_evaluation, run_lambda_sweep,
                           plot_phase_space_lambda_sweep,
                           plot_lambda_sweep_metrics)


def sanity_check(device):
    """Forward-pass and loss finiteness check for both models."""
    print("\n── Sanity check ──")
    from losses import total_hdae_loss, reconstruction_loss

    dummy = torch.randn(2, 1, 1792).to(device)

    hae = HAE(latent_dim=2).to(device)
    x_hat, q, p, H_t = hae(dummy)
    assert x_hat.shape == dummy.shape
    loss, recon, phys = total_hdae_loss(x_hat, dummy, H_t, lambda_phys=0.1)
    assert torch.isfinite(loss), f"H-AE loss not finite: {loss}"

    cae = BaselineCAE(latent_dim=2).to(device)
    x_hat2, z = cae(dummy)
    r2 = reconstruction_loss(x_hat2, dummy)
    assert torch.isfinite(r2), f"CAE loss not finite: {r2}"

    loss.backward()   # verify backward through H-AE

    print(f"  H-AE: loss={loss.item():.4f} (recon={recon.item():.4f}, "
          f"phys={phys.item():.6f})")
    print(f"  CAE : loss={r2.item():.4f}")
    print(f"  Encoder output: {hae.encoder(dummy).shape}  "
          f"(expected (2, 4, 112))")
    print("Sanity check passed.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────────────────

def stage_hparam_search(loaders, device):
    best_hae_cfg, best_cae_cfg, best_properhae_cfg, best_porthae_cfg = run_hparam_search(
        loaders, device, results_dir=RESULTS_DIR, seed=SEED
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BEST_CFG_PATH, "w") as f:
        json.dump({
            "hae":       best_hae_cfg,
            "cae":       best_cae_cfg,
            "properhae": best_properhae_cfg,
            "porthae":   best_porthae_cfg,
        }, f, indent=2)
    print(f"Best configs saved to {BEST_CFG_PATH}")
    return best_hae_cfg, best_cae_cfg, best_properhae_cfg, best_porthae_cfg


def stage_train(loaders, device, hae_cfg, cae_cfg, properhae_cfg, porthae_cfg):
    set_seeds(SEED)
    hae = HAE(latent_dim=hae_cfg["latent_dim"]).to(device)
    print(f"\n── Final training — HAE ──  config: {hae_cfg}")
    hae_history, _ = train_model(
        hae, loaders, device,
        model_name     = "hae",
        checkpoint_dir = CHECKPOINT_DIR,
        log_path       = LOG_PATH,
        lambda_phys    = hae_cfg["lambda_phys"],
        lr             = hae_cfg["lr"],
        max_epochs     = MAX_EPOCHS,
        patience       = PATIENCE,
    )

    set_seeds(SEED)
    cae = BaselineCAE(latent_dim=cae_cfg["latent_dim"]).to(device)
    print(f"\n── Final training — CAE ──  config: {cae_cfg}")
    cae_history, _ = train_model(
        cae, loaders, device,
        model_name     = "cae",
        checkpoint_dir = CHECKPOINT_DIR,
        log_path       = LOG_PATH,
        lambda_phys    = 0.0,
        lr             = cae_cfg["lr"],
        max_epochs     = MAX_EPOCHS,
        patience       = PATIENCE,
    )

    set_seeds(SEED)
    properhae = ProperHAE(latent_dim=properhae_cfg["latent_dim"]).to(device)
    print(f"\n── Final training — ProperHAE ──  config: {properhae_cfg}")
    properhae_history, _ = train_model(
        properhae, loaders, device,
        model_name     = "properhae",
        checkpoint_dir = CHECKPOINT_DIR,
        log_path       = LOG_PATH,
        lambda_phys    = properhae_cfg["lambda_phys"],
        lr             = properhae_cfg["lr"],
        max_epochs     = MAX_EPOCHS,
        patience       = PATIENCE,
    )

    set_seeds(SEED)
    porthae = PortHAE(latent_dim=porthae_cfg["latent_dim"]).to(device)
    print(f"\n── Final training — PortHAE ──  config: {porthae_cfg}")
    porthae_history, _ = train_model(
        porthae, loaders, device,
        model_name     = "porthae",
        checkpoint_dir = CHECKPOINT_DIR,
        log_path       = LOG_PATH,
        lambda_phys    = porthae_cfg["lambda_phys"],
        lr             = porthae_cfg["lr"],
        max_epochs     = MAX_EPOCHS,
        patience       = PATIENCE,
    )

    return (hae, cae, properhae, porthae,
            hae_history, cae_history, properhae_history, porthae_history)


def stage_evaluate(loaders, info, device,
                   hae_cfg, cae_cfg, properhae_cfg, porthae_cfg,
                   hae_history, cae_history):
    """Load best checkpoints for all models and run all evaluation experiments."""
    model_specs = [
        ("hae",       HAE,         hae_cfg["latent_dim"]),
        ("cae",       BaselineCAE, cae_cfg["latent_dim"]),
        ("properhae", ProperHAE,   properhae_cfg["latent_dim"]),
        ("porthae",   PortHAE,     porthae_cfg["latent_dim"]),
    ]
    loaded = {}
    for name, cls, ld in model_specs:
        m    = cls(latent_dim=ld).to(device)
        ckpt = os.path.join(CHECKPOINT_DIR, f"{name}_best.pt")
        if os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"Loaded {name} weights from {ckpt}")
        else:
            print(f"  WARNING: no checkpoint at {ckpt} — using random weights")
        loaded[name] = m

    return run_evaluation(
        loaded["hae"], loaded["cae"], loaders, info, device,
        hae_history    = hae_history,
        cae_history    = cae_history,
        physics_models = {"properhae": loaded["properhae"],
                          "porthae":   loaded["porthae"]},
        plots_dir      = PLOTS_DIR,
        results_dir    = RESULTS_DIR,
    )


def stage_physics_variants(loaders, device, hae_cfg, cae_cfg):
    """Train ProperHAE and PortHAE across lambda values, plot loss curves and phase space."""
    from run_physics_variants import run_physics_variants
    print("\n── Physics variants (ProperHAE + PortHAE) ──")

    # Load CAE as reference for phase-space comparison
    cae = BaselineCAE(latent_dim=cae_cfg["latent_dim"]).to(device)
    ckpt = os.path.join(CHECKPOINT_DIR, "cae_best.pt")
    if os.path.exists(ckpt):
        cae.load_state_dict(torch.load(ckpt, map_location=device))
        cae.eval()
    else:
        print("  CAE checkpoint not found — phase space will omit reference column.")
        cae = None

    return run_physics_variants(
        loaders, device,
        latent_dim     = hae_cfg["latent_dim"],
        lr             = hae_cfg["lr"],
        checkpoint_dir = CHECKPOINT_DIR,
        log_path       = LOG_PATH,
        plots_dir      = PLOTS_DIR,
        max_epochs     = MAX_EPOCHS,
        patience       = PATIENCE,
        cae            = cae,
    )


def stage_lambda_sweep(loaders, info, device, hae_cfg, cae_cfg,
                       base_metrics=None):
    """
    Train H-AE at λ∈{0.01, 0.1, 1.0} using best latent_dim and lr,
    compute metrics, generate the two lambda-sweep plots, and merge
    results into final_metrics.json.
    """
    sweep_results, models_dict = run_lambda_sweep(
        loaders, device,
        latent_dim     = hae_cfg["latent_dim"],
        lr             = hae_cfg["lr"],
        checkpoint_dir = CHECKPOINT_DIR,
        plots_dir      = PLOTS_DIR,
        results_dir    = RESULTS_DIR,
        max_epochs     = MAX_EPOCHS,
        patience       = PATIENCE,
        seed           = SEED,
    )

    # Load CAE for phase-space comparison
    cae = BaselineCAE(latent_dim=cae_cfg["latent_dim"]).to(device)
    ckpt = os.path.join(CHECKPOINT_DIR, "cae_best.pt")
    if os.path.exists(ckpt):
        cae.load_state_dict(torch.load(ckpt, map_location=device))

    test_loader = loaders.get("test1") or loaders.get("test")
    plot_phase_space_lambda_sweep(
        models_dict, cae, test_loader, device, PLOTS_DIR
    )
    plot_lambda_sweep_metrics(
        sweep_results,
        cae_metrics = base_metrics or {},
        plots_dir   = PLOTS_DIR,
    )

    # Merge into final_metrics.json
    json_path = os.path.join(RESULTS_DIR, "final_metrics.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    all_metrics["lambda_sweep"] = sweep_results
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Lambda sweep results merged into {json_path}")

    return sweep_results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hamiltonian Autoencoder for BBH GW inspiral signals"
    )
    parser.add_argument("--run-all",       action="store_true")
    parser.add_argument("--hparam-search", action="store_true")
    parser.add_argument("--train",         action="store_true")
    parser.add_argument("--evaluate",      action="store_true")
    parser.add_argument("--lambda-sweep",      action="store_true",
                        help="Run lambda sweep (can combine with --evaluate)")
    parser.add_argument("--physics-variants", action="store_true",
                        help="Train ProperHAE + PortHAE and plot their loss curves")
    parser.add_argument("--no-hparam",     action="store_true",
                        help="Skip hparam search and use default configs")
    parser.add_argument("--batch-size",    type=int, default=BATCH_SIZE)
    parser.add_argument("--hdf",           type=str, default=HDF_PATH)
    args = parser.parse_args()

    if not any([args.run_all, args.hparam_search, args.train,
                args.evaluate, args.lambda_sweep, args.physics_variants]):
        args.run_all = True

    print("=" * 60)
    print("Hamiltonian Autoencoder — BBH Inspiral Signal Learning")
    print("=" * 60)

    set_seeds(SEED)
    device = get_device()
    sanity_check(device)

    print(f"\nLoading data from: {args.hdf}")
    loaders, info = make_splits(args.hdf, batch_size=args.batch_size)
    print(f"Batch size in use: {info['batch_size']}")

    hae_cfg       = DEFAULT_HAE_CFG
    cae_cfg       = DEFAULT_CAE_CFG
    properhae_cfg = DEFAULT_PROPERHAE_CFG
    porthae_cfg   = DEFAULT_PORTHAE_CFG

    # ── Hparam search ──────────────────────────────────────────────────────
    if args.run_all or args.hparam_search:
        if not args.no_hparam:
            set_seeds(SEED)
            hae_cfg, cae_cfg, properhae_cfg, porthae_cfg = \
                stage_hparam_search(loaders, device)
        else:
            print(f"\nUsing default configs")

    def _load_saved_configs():
        nonlocal hae_cfg, cae_cfg, properhae_cfg, porthae_cfg
        if os.path.exists(BEST_CFG_PATH):
            with open(BEST_CFG_PATH) as f:
                cfgs = json.load(f)
            hae_cfg       = cfgs.get("hae",       hae_cfg)
            cae_cfg       = cfgs.get("cae",        cae_cfg)
            properhae_cfg = cfgs.get("properhae",  properhae_cfg)
            porthae_cfg   = cfgs.get("porthae",    porthae_cfg)
            print(f"Loaded configs from {BEST_CFG_PATH}")
        else:
            print("No saved configs found — using defaults.")

    if (args.train or args.evaluate or args.physics_variants) and not args.run_all:
        _load_saved_configs()

    # ── Training ───────────────────────────────────────────────────────────
    hae_history = cae_history = None
    if args.run_all or args.train:
        set_seeds(SEED)
        (_, _, _, _,
         hae_history, cae_history, _, _) = stage_train(
            loaders, device, hae_cfg, cae_cfg, properhae_cfg, porthae_cfg)

    # ── Evaluation ─────────────────────────────────────────────────────────
    base_metrics = None
    if args.run_all or args.evaluate:
        if hae_history is None:
            hae_history = {"train_recon": [], "val_recon": [],
                           "train_phys": [], "val_phys": []}
            cae_history = {"train_recon": [], "val_recon": [],
                           "train_phys": [], "val_phys": []}
        base_metrics = stage_evaluate(
            loaders, info, device,
            hae_cfg, cae_cfg, properhae_cfg, porthae_cfg,
            hae_history, cae_history)

    # ── Lambda sweep ───────────────────────────────────────────────────────
    if args.run_all or args.lambda_sweep:
        if args.lambda_sweep and not args.run_all:
            _load_saved_configs()
        set_seeds(SEED)
        stage_lambda_sweep(loaders, info, device, hae_cfg, cae_cfg, base_metrics)

    # ── Physics variants (λ sweep for analysis + phase space plots) ────────
    if args.run_all or args.physics_variants:
        if args.physics_variants and not args.run_all:
            _load_saved_configs()
        set_seeds(SEED)
        stage_physics_variants(loaders, device, hae_cfg, cae_cfg)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
