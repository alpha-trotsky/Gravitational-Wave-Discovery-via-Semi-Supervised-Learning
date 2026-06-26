# `src/` — Source Overview

## Architecture in One Paragraph

The pipeline trains convolutional autoencoders on clean BBH inspiral signals
(`h1_signal`) to learn a compact latent space, then uses that manifold for
signal detection and denoising. Four model variants share the same CNN
encoder/decoder backbone: a vanilla **CAE** baseline, a variance-based
**HAE/VHAE** (penalises time-variance of H), a strictly conservative
**ProperHAE** (enforces `ż = J∇H`), and a dissipative **PortHAE** (enforces
`ż = (J−R)∇H`). Train/val/test splits are made by progenitor mass so
the held-out test sets are OOD in mass. Evaluation measures reconstruction
MSE, waveform overlap, latent PC1 ratio, noise robustness, and sweeps
over λ (physics loss weight).

---

## Directory Structure

```
src/
├── core/                   Shared model, data, and training code
│   ├── data.py
│   ├── models.py
│   ├── losses.py
│   └── train.py
├── training/               Kaggle-side: hparam search + full training
│   ├── config.py
│   ├── hparam_hae.py
│   ├── hparam_physics.py
│   └── train_all.py
├── evaluation/             Local-side: one script per experiment
│   ├── config.py
│   ├── utils.py
│   ├── reconstruction.py
│   ├── phase_space.py
│   ├── ood_generalization.py
│   ├── noise_robustness.py
│   ├── lambda_sweep.py
│   ├── mlp_param_estimator.py
│   └── compare_models.py
└── smoke_test.py
```

**Import convention:** every non-core script adds `src/` to `sys.path` via
`sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`,
so imports take the form `from core.models import ...` and
`from evaluation.utils import ...`.

**Checkpoint layout** produced by `train_all.py`:
```
checkpoints/
├── hae/        lambda0.1_best.pt  lambda0.25_best.pt  lambda0.5_best.pt  lambda1.0_best.pt
├── cae/        best.pt
├── properhae/  lambda0.1_best.pt  ...
└── porthae/    lambda0.1_best.pt  ...
```

---

## `core/data.py`

Loads `output/dataset.hdf`, crops signals to a 1 792-sample inspiral window,
normalises by the standard deviation of each sample's centre crop, and builds
train/val/test `DataLoader`s.

**Key constants**

| Symbol | Value | Meaning |
|--------|-------|---------|
| `EVENT_IDX` | 11 264 | Merger sample index (5.5 s × 2 048 Hz) |
| `CROP_LEN`  | 1 792  | Final sample length fed to models |
| `MAX_JITTER`| 256    | Training-time random time-shift (± 125 ms) |
| `WIDE_LEN`  | 2 304  | Stored window jitter draws from |

**Mass-based OOD splits**
- Train: `mass1 ≤ 65 & mass2 ≤ 65` (excluding test1 overlap), 15% holdout → val
- Test1: `mass2 ∈ [5,15] & mass1 ∈ [50,65]` — asymmetric intermediate mass-ratio
- Test2: `mass2 ≤ 20 & mass1 > 65` — extreme mass-ratio, heavy primary

`make_splits(hdf_path, batch_size)` returns `(loaders_dict, info_dict)`.
`loaders_dict` keys: `train`, `train_eval`, `val`, `test1`, `test2`.
`info_dict` keys: `train_idx`, `val_idx`, `test1_idx`, `test2_idx`, `hdf_path`, etc.

`GWDataset` stores the wide window and slices the final `CROP_LEN` samples at
`__getitem__` time with random jitter (train) or fixed centre crop (val/test).

---

## `core/models.py`

Defines the encoder, decoder, Hamiltonian MLP, all four model classes, and the
two physics loss *functions* that require second-order autograd.

**Backbone**
```
CNNEncoder : (B, 1, 1792) → Conv1d ×4 (strides 2) → 1×1 proj → (B, 2·D, 112)
CNNDecoder : (B, 2·D, 112) → 1×1 proj → ConvTranspose1d ×4 → (B, 1, 1792)
HamiltonianMLP : R^(2D) → Linear(64)→Tanh → Linear(64)→Tanh → Linear(1)
```
`D = latent_dim` (default 2). Latent shape `(B, 4, 112)`.

**Model classes**

| Class | `forward()` returns | `physics_type` | Loss |
|-------|---------------------|----------------|------|
| `BaselineCAE` | `(x_hat, z)` | — | MSE only |
| `VHAE` / `HAE` | `(x_hat, q, p, H_t)` | `'variance'` | `recon + λ·var(H_t)` |
| `ProperHAE` | `(x_hat, q, p, None)` | `'proper'` | `recon + λ·MSE(ż, J∇H)` |
| `PortHAE`   | `(x_hat, q, p, None)` | `'port'`   | `recon + λ·MSE(ż, (J−R)∇H)` |

`HAE` is a backwards-compatible alias for `VHAE`. Latent split:
`q = z[:, :D, :]`, `p = z[:, D:, :]`. `physics_type` is set in `__init__`.

**Physics loss functions**

`proper_hnn_loss(hamiltonian_net, z, dt, create_graph)` — enforces
conservative mechanics. Computes `ż_actual` via central finite differences,
`∇_z H` via `torch.autograd.grad`, then `ż_pred = J ∇H`. Loss = MSE.
`create_graph=True` required during training; `False` during validation.

`port_hamiltonian_loss(hamiltonian_net, R_matrix, z, dt, create_graph)` —
same structure but `ż_pred = (J − R) ∇H` where `R = L Lᵀ` is a learned
PSD dissipation matrix stored in `PortHAE.L_matrix`.

`ProperHAE` and `PortHAE` return `H_t = None` from `forward()` — the
Hamiltonian MLP is called only once, inside the loss function.

---

## `core/losses.py`

Lightweight losses for the variance-based HAE:

| Function | Formula |
|----------|---------|
| `reconstruction_loss(x_hat, target)` | `F.mse_loss` |
| `hamiltonian_loss(H_t)` | `mean_batch(var_time(H_t))` |
| `total_hdae_loss(x_hat, target, H_t, λ)` | `recon + λ·phys` → `(total, recon, phys)` |

No second-order autograd needed here.

---

## `core/train.py`

**`_train_epoch(model, loader, optimizer, device, lambda_phys)`**
Dispatches on `model.physics_type`:
- `variance` / CAE → `total_hdae_loss` (no second-order grad)
- `proper` → `proper_hnn_loss(create_graph=True)`
- `port`   → `port_hamiltonian_loss(create_graph=True)`

Gradient clipping at `max_norm=1.0` before every `optimizer.step()`.

**`_val_epoch(model, loader, device, lambda_phys)`**
Mirror of train epoch but `model(x)` is wrapped in `torch.no_grad()` for
`variance`/CAE branches. For `proper`/`port`, `model(x)` runs outside
no_grad (needed to build the graph for `autograd.grad`), then the physics
metric is computed inside `torch.enable_grad()` with `create_graph=False`.

**`train_model(...)`** — Adam optimiser, early stopping on val reconstruction
MSE. Saves best weights to `{checkpoint_dir}/{model_name}_best.pt`.

**`quick_train(...)`** — Capped-epoch version for hparam search (no
checkpointing, no logging). Returns best val MSE.

---

## `training/config.py`

Single source of truth for all training hyperparameters and paths. Edit this
before running on Kaggle.

| Constant | Default | Purpose |
|----------|---------|---------|
| `LAMBDA_VALUES` | `[0.1, 0.25, 0.5, 1.0]` | Lambda sweep for all physics models |
| `HAE_LR_GRID` | `[1e-4, 1e-3]` | LR search space for HAE/CAE |
| `PHYSICS_LR_GRID` | `[5e-5, 1e-4, 5e-4]` | LR search space for ProperHAE/PortHAE |
| `LATENT_DIM_GRID` | `[1, 2, 4]` | Shared latent dim search space |
| `HPARAM_LAMBDA_FIXED` | `0.1` | Lambda fixed during hparam search |
| `MAX_EPOCHS` | `500` | Hard cap for full training |
| `PATIENCE` | `200` | Early stopping patience |

---

## `training/hparam_hae.py`

Grid search over `lr × latent_dim` for HAE at `lambda=HPARAM_LAMBDA_FIXED`.
Uses `quick_train` (50 epochs, patience 10). Writes best `{latent_dim, lr}`
to `results/best_configs.json` under keys `"hae"` and `"cae"` (CAE mirrors HAE).

**Why HAE → apply to CAE:** both share identical backbone; HAE is the harder
problem (extra physics constraint), so its optimal architecture is a conservative
lower bound for CAE.

---

## `training/hparam_physics.py`

Same structure but for `ProperHAE` with a lower LR grid (`PHYSICS_LR_GRID`).
Writes best config to `best_configs.json` under `"properhae"` and `"porthae"`.

**Why lower LR grid:** `torch.autograd.grad` through the Hamiltonian MLP
produces larger gradient magnitudes than the variance-based loss, making
higher learning rates unstable.

---

## `training/train_all.py`

Trains all 13 model checkpoints in sequence:
- HAE × 4 lambdas → `checkpoints/hae/lambda{x}_best.pt`
- CAE × 1         → `checkpoints/cae/best.pt`
- ProperHAE × 4   → `checkpoints/properhae/lambda{x}_best.pt`
- PortHAE × 4     → `checkpoints/porthae/lambda{x}_best.pt`

Reads `best_configs.json` for `latent_dim` and `lr`; falls back to
`DEFAULT_HAE_CFG` / `DEFAULT_PHYSICS_CFG` if the file is absent.
Accepts `--models hae cae` to train a subset (useful for resuming after crash).
Saves all training histories to `results/training_histories.json` for loss
curve plots.

---

## `evaluation/config.py`

Defines `CHECKPOINT_DIR`, `PLOTS_DIR`, `RESULTS_DIR`, `LAMBDA_VALUES`.

`build_model_registry(checkpoint_dir, results_dir, lambda_values)` returns a
list of `(display_name, cls, latent_dim, ckpt_path)` tuples for every model ×
lambda combination. Reads `latent_dim` from `best_configs.json`.

`ckpt_path(model_family, lam, checkpoint_dir)` returns the checkpoint path
for a given family and lambda.

---

## `evaluation/utils.py`

Shared inference and metric utilities imported by all eval scripts:

| Function | Purpose |
|----------|---------|
| `load_model(cls, latent_dim, ckpt_path, device)` | Instantiate + load weights; `None` if missing |
| `load_registry(registry, device)` | Load all checkpoints from a registry list |
| `collect_outputs(model, loader, device)` | `→ (x, y, x_hat, z)` CPU tensors |
| `infer_batch(model, x_tensor, device)` | Inference on a plain tensor |
| `mse(a, b)` | Scalar MSE |
| `overlap(x_hat, y)` | Mean cosine similarity |
| `compute_pc1_ratio(model, loader, device)` | Avg fraction of latent variance in PC1 |
| `compute_metrics(model, loader, device)` | `→ {mse, overlap, pc1}` dict |

---

## Evaluation Scripts

Each script is fully standalone: loads data, loads available checkpoints,
skips missing ones, produces output. No dependency on training scripts.

| Script | Produces |
|--------|---------|
| `reconstruction.py` | Per-model MSE/overlap/PC1 on test1+test2; waveform plots |
| `phase_space.py` | Latent trajectory scatter plots (q₀ vs p₀ or z₀ vs z₁) |
| `ood_generalization.py` | Train vs test MSE bar chart; gap table |
| `noise_robustness.py` | MSE/overlap when noisy h1_strain fed to clean-trained model |
| `lambda_sweep.py` | MSE/overlap/PC1 vs λ per model family; loads pre-trained ckpts |
| `compare_models.py` | Ranked comparison table across all models × test sets |
| `mlp_param_estimator.py` | MLP probing of frozen latent → [mass1, mass2, spin1z, spin2z] |

---

## Data Flow

```
dataset.hdf
   ├── injection_parameters/h1_signal  →  GWDataset  →  train/val/test1/test2 loaders
   └── injection_samples/h1_strain     →  load_noisy_test_data (noise_robustness.py)

training/hparam_hae.py      → results/best_configs.json  (hae, cae entries)
training/hparam_physics.py  → results/best_configs.json  (properhae, porthae entries)

training/train_all.py
   ├── reads best_configs.json
   ├── trains 13 models via core/train.py
   └── checkpoints/{family}/lambda{x}_best.pt
       results/training_histories.json

evaluation/*.py
   ├── read checkpoints/ (skip missing)
   ├── run experiment via core/
   └── plots/ + results/*.json
```
