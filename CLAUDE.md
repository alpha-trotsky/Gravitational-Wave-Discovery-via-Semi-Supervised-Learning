# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physically-inspired deep learning for gravitational wave detection. Convolutional autoencoders are trained **only on clean synthetic BBH inspiral waveforms** (`h1_signal`), with Hamiltonian physics encoded into the loss to structure the latent space. The trained model then generalises to held-out mass ranges and denoises real whitened LIGO strain (`h1_strain`) — no labelled noise data used during training.

## Commands

**Smoke test (no HDF file required — run this first):**
```bash
python src/smoke_test.py
```

**On Kaggle — hparam search then train all models:**
```bash
python src/training/hparam_hae.py        # HAE grid search → saves hae/cae best config
python src/training/hparam_physics.py    # ProperHAE grid search → saves properhae/porthae best config
python src/training/train_all.py         # train all 13 checkpoints
python src/training/train_all.py --models hae cae   # train a subset only
```

**Locally — each evaluation script is standalone:**
```bash
python src/evaluation/reconstruction.py
python src/evaluation/phase_space.py
python src/evaluation/ood_generalization.py
python src/evaluation/noise_robustness.py
python src/evaluation/lambda_sweep.py
python src/evaluation/compare_models.py
python src/evaluation/mlp_param_estimator.py
```

**Optional flags (all scripts accept these):**
```
--hdf ./output/dataset.hdf
--checkpoint-dir ./checkpoints
--batch-size 16
```

## Key Dependencies

No requirements file. Core deps: `torch`, `h5py`, `numpy`, `matplotlib`, `pandas`, `tqdm`.

## Architecture

### Directory Structure

```
src/
├── core/           Shared model, data, training code — imported by everything
├── training/       Kaggle-side: hparam search + full training
├── evaluation/     Local-side: one script per experiment
└── smoke_test.py
```

**Import convention:** `training/` and `evaluation/` scripts add `src/` to `sys.path` at startup, so imports are `from core.models import ...`, `from evaluation.utils import ...`.

**Checkpoint layout:**
```
checkpoints/hae/        lambda0.1_best.pt  lambda0.25_best.pt  lambda0.5_best.pt  lambda1.0_best.pt
checkpoints/cae/        best.pt
checkpoints/properhae/  lambda0.1_best.pt  ...
checkpoints/porthae/    lambda0.1_best.pt  ...
```

### `core/data.py`

HDF5 at `./output/dataset.hdf`:
- `injection_parameters/h1_signal` — `(N, 16384)` clean BBH waveforms
- `injection_parameters/mass1`, `mass2` — used for OOD split
- `injection_samples/h1_strain` — whitened noisy strain (eval only)

Key constants: `SAMPLING_RATE=2048`, `EVENT_IDX=11264` (merger at 5.5 s), `CROP_LEN=1792`, `MAX_JITTER=256`.

**Mass-based OOD splits:**
- Train: `mass1≤65 & mass2≤65` (excluding test1 overlap), 15% random holdout → val
- Test1: `mass2∈[5,15] & mass1∈[50,65]` — asymmetric intermediate mass-ratio
- Test2: `mass2≤20 & mass1>65` — extreme mass-ratio, heavy primary

`make_splits(hdf_path, batch_size)` returns `(loaders_dict, info_dict)` where `loaders_dict` has keys `train`, `train_eval`, `val`, `test1`, `test2` and `info_dict` contains index arrays.

### `core/models.py`

All four variants share `CNNEncoder` → `CNNDecoder` backbone (`(B,1,1792) → (B,2D,112)` where D=`latent_dim`, default 2) and a `HamiltonianMLP` (2D→64→64→1).

| Class | `forward()` returns | `physics_type` | Loss |
|-------|---------------------|----------------|------|
| `BaselineCAE` | `(x_hat, z)` | — | MSE only |
| `VHAE` / `HAE` | `(x_hat, q, p, H_t)` | `'variance'` | `recon + λ·var(H_t)` |
| `ProperHAE` | `(x_hat, q, p, None)` | `'proper'` | `recon + λ·MSE(ż, J∇H)` |
| `PortHAE` | `(x_hat, q, p, None)` | `'port'` | `recon + λ·MSE(ż, (J−R)∇H)` |

`HAE` is a backwards-compatible alias for `VHAE`. `physics_type` is set in `__init__`. `ProperHAE`/`PortHAE` return `H_t=None` — the Hamiltonian MLP is called only once, inside the physics loss function. Latent split: `q = z[:, :D, :]`, `p = z[:, D:, :]`.

`ProperHAE`/`PortHAE` use second-order autograd (`create_graph=True` during training, `False` during validation). `_val_epoch` calls `model(x)` outside `no_grad` for these branches, then computes the physics metric inside `torch.enable_grad()`.

### `training/config.py`

Single source of truth for all training hyperparameters. Edit before running on Kaggle.

| Constant | Default | Purpose |
|----------|---------|---------|
| `LAMBDA_VALUES` | `[0.1, 0.25, 0.5, 1.0]` | Lambda sweep for all physics models |
| `HAE_LR_GRID` | `[1e-4, 1e-3]` | LR search space for HAE/CAE |
| `PHYSICS_LR_GRID` | `[5e-5, 1e-4, 5e-4]` | LR search space for ProperHAE/PortHAE (skewed lower due to second-order gradient noise) |
| `LATENT_DIM_GRID` | `[1, 2, 4]` | Shared latent dim search space |
| `HPARAM_LAMBDA_FIXED` | `0.1` | Lambda fixed during hparam search |
| `MAX_EPOCHS` / `PATIENCE` | `500` / `200` | Full training run limits |

### `evaluation/config.py` and `evaluation/utils.py`

`build_model_registry()` returns the full list of `(name, cls, latent_dim, ckpt_path)` for all 13 models. `load_registry(registry, device)` loads all available checkpoints and skips missing ones — every eval script uses these two functions to stay robust to partially-complete training runs.

`evaluation/utils.py` exports: `collect_outputs`, `infer_batch`, `mse`, `overlap`, `compute_pc1_ratio`, `compute_metrics`, `load_model`.
