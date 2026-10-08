# Model Architectures and Results So Far

Companion to [`SOURCE_OVERVIEW.md`](SOURCE_OVERVIEW.md). That file describes
*where* the code lives; this one describes *what the models do*, the physics
behind each variant, and every result produced to date.

> **Status (2026-09-25):** all results below come from the preliminary
> training run of **2026-06-24** (max 200 epochs, early-stopping patience 20).
> The full run specified in `training/config.py` (500 epochs, patience 200,
> λ ∈ {0.1, 0.25, 0.5, 1.0}) has **not** been done yet. See
> [Caveats](#caveats-on-the-current-results) before quoting any number.

---

## 1. Shared Backbone

All four variants use the same encoder and decoder, so any difference between
them comes only from the physics term in the loss.

### Encoder: `CNNEncoder`

Input: a 1 792-sample crop of the whitened waveform (≈ 0.875 s at 2 048 Hz,
ending near merger). It passes through four strided `Conv1d → GroupNorm → GELU`
blocks, each halving the time axis:

```
(B,1,1792) → 16ch/896 → 32ch/448 → 64ch/224 → 128ch/112 → 1×1 conv → (B, 2D, 112)
```

The output is a **latent trajectory**, not a single vector: 112 latent time
steps, each a point in a 2D-dimensional phase space (`D = latent_dim`). One
latent step covers 16 input samples (≈ 7.8 ms). The channels split into
generalised position and momentum:

```
q = z[:, :D, :]      p = z[:, D:, :]
```

### Decoder: `CNNDecoder`

The mirror of the encoder: a 1×1 conv, then `ConvTranspose1d` blocks (with
`output_padding=1` for exact 2× upsampling), giving `(B,1,1792)`.

### Hamiltonian network: `HamiltonianMLP`

```
H : R^(2D) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(1)
```

A learned scalar "energy" over phase space. It is applied to each of the
112 latent time steps independently. Tanh keeps H smooth, which matters
because ProperHAE and PortHAE differentiate it.

---

## 2. The Four Variants

### 2.1 `BaselineCAE`: control

The encoder and decoder only, with no Hamiltonian and no physics term.

```
L = MSE(x̂, x)
```

This shows what the latent space looks like with no physics prior.
`forward()` returns `(x_hat, z)`.

### 2.2 `VHAE` / `HAE`: variance-conserving (weak formulation)

H is evaluated at every latent step, giving `H_t ∈ R^(B×112)`. The physics
term penalises how much H changes over time:

```
L = MSE(x̂, x) + λ · mean_b Var_t(H_t)
```

This asks for "energy that stays roughly constant along the trajectory". It is
cheap because no second-order gradients are needed. However, it **never says
how z should move**. It only asks that H be flat along whatever path the
encoder produces.

### 2.3 `ProperHAE`: true Hamiltonian dynamics (conservative)

This enforces Hamilton's equations directly on the latent trajectory:

```
ż = J ∇_z H,      J = [[0, I], [−I, 0]]
⇔  q̇ = ∂H/∂p,     ṗ = −∂H/∂q
```

How `proper_hnn_loss` computes it:

1. **Actual velocity.** Central finite differences along the encoder's
   trajectory (`compute_latent_velocity`, dt = 1 latent step; one-sided
   differences at the two ends).
2. **Predicted velocity.** Run H on every `(q, p)` point, take `∇_z H` via
   `torch.autograd.grad`, and multiply by J.
3. **Loss.** `MSE(ż_actual, J∇H)`.

Because J is skew-symmetric, `dH/dt = ∇Hᵀ J ∇H = 0`: energy conservation
follows from the dynamics instead of being imposed as a separate penalty.
Getting the gradient back to the encoder through `∇H` needs
`create_graph=True` (second-order autograd), which makes training slower and
noisier. That is why `PHYSICS_LR_GRID` is skewed toward lower learning rates.

### 2.4 `PortHAE`: port-Hamiltonian (dissipative)

This extends ProperHAE with a learned dissipation matrix R:

```
ż = (J − R) ∇_z H,      R = L Lᵀ  ⪰ 0
```

- `L` is a learnable `(2D × 2D)` parameter (`PortHAE.L_matrix`), initialised
  to `0.1·I`. Writing R as `L Lᵀ` keeps it positive semi-definite.
- Energy now decreases monotonically:

  ```
  dH/dt = ∇Hᵀ (J − R) ∇H = −∇Hᵀ R ∇H ≤ 0
  ```

- **Physical reason.** A BBH inspiral is not a conservative system. The binary
  loses orbital energy to gravitational radiation, and that loss drives the
  chirp. ProperHAE forces a closed, energy-conserving latent orbit; PortHAE
  lets the latent trajectory lose energy at a learned rate, which is closer
  to the real physics.
- R is a *constant* matrix: the dissipation rate cannot depend on the state.

### 2.5 Summary

| Model | `physics_type` | Constraint on latent dynamics | Energy behaviour | Cost |
|---|---|---|---|---|
| CAE | — | none | — | cheap |
| VHAE / HAE | `variance` | `Var_t(H) → 0` | roughly constant (soft) | cheap |
| ProperHAE | `proper` | `ż = J∇H` | conserved by the dynamics | 2nd-order grad |
| PortHAE | `port` | `ż = (J−R)∇H`, `R ⪰ 0` | monotonically dissipated | 2nd-order grad + R |

All physics terms are **soft penalties** weighted by λ, not hard
architectural constraints.

### 2.6 Known failure modes of the physics terms

These are ways each term can go to zero without doing what it is meant to:

- **VHAE:** the MLP alone can make H constant (last-layer output → constant),
  which drives `Var_t(H)` to 0 while the encoder feels no physics pressure at
  all.
- **ProperHAE / PortHAE:** if H becomes flat, `∇H ≈ 0` and the predicted
  velocity is 0. The loss then reduces to `‖ż‖²`, which is just a smoothness
  penalty on the latent trajectory.
- **PortHAE:** if R → 0 it falls back to ProperHAE behaviour. Checking the
  eigenvalues of `model.get_R()` shows whether this has happened (see §4.4).

---

## 3. Experimental Setup (as actually run)

- **Data:** `output/dataset.hdf`. Trained only on clean `h1_signal`, normalised
  per sample by the std of the centre crop, with ±256-sample random time jitter
  during training.
- **Splits (by progenitor mass, so the test sets are out of distribution):**
  - Train: `m1 ≤ 65 & m2 ≤ 65` (excluding test1), with 15 % held out as val
  - Test1: `m2 ∈ [5,15] & m1 ∈ [50,65]`: asymmetric, intermediate mass ratio
  - Test2: `m2 ≤ 20 & m1 > 65`: extreme mass ratio, heavy primary
- **Metrics:** MSE, overlap (mean cosine similarity between x̂ and y), and
  PC1 ratio (average fraction of per-sample latent variance captured by the
  first principal component; higher means a more one-dimensional, less
  structured trajectory).
- **Training run of 2026-06-24:** Adam, lr = 1e-3 for every model, max 200
  epochs, patience 20, batch size 16, seed 42.

### Hyperparameter search (`results_not_trained_enough/results/hparam_search.csv`)

Grid: λ ∈ {0.01, 0.1, 1.0} × latent_dim ∈ {1, 2, 4} × lr ∈ {1e-4, 1e-3},
quick-train mode. Best validation MSE per family:

| Family | Best config | Best val MSE |
|---|---|---:|
| CAE | D=2, lr=1e-3 | 1.41e-4 |
| HAE | D=2, lr=1e-3, λ=0.01 | 1.44e-4 |
| ProperHAE | D=4, lr=1e-3, λ=0.01 | **1.23e-4** |
| PortHAE | D=2, lr=1e-3, λ=0.01 | 1.84e-4 |

What the grid showed:

- lr = 1e-3 beat 1e-4 almost everywhere in this short-budget regime. That
  conflicts with the current `PHYSICS_LR_GRID = [5e-5, 1e-4, 5e-4]`, which no
  longer contains the learning rate that won.
- For the physics models, larger λ hurts badly at D=1–2 (for example ProperHAE
  at λ=1, D=1, lr=1e-4 gave 1.2e-3) but only mildly at D=4. A larger latent
  space absorbs the physics constraint more easily.
- λ has only a weak, non-monotonic effect on HAE: for most configs, changing λ
  across two orders of magnitude moves val MSE by < 5 %. The exception is the
  best config (D=2, lr=1e-3), which ranges from 1.4e-4 to 2.2e-4. This fits
  the VHAE degeneracy in §2.6, although single-seed noise cannot be ruled out.

The current `results/best_configs.json` stores latent_dim only: CAE/HAE D=2,
ProperHAE/PortHAE D=4. Every ProperHAE and PortHAE checkpoint in
`checkpoints/` has D=4 (8 latent channels).

---

## 4. Results

### 4.1 Training runs (`results_not_trained_enough/results/training_log.txt`)

| Run | λ | Stopped at epoch | Best val MSE |
|---|---:|---:|---:|
| CAE | — | 174 | **2.5e-5** |
| HAE | 0.01 | 64 | 1.44e-4 |
| ProperHAE | 0.01 | 62 | 1.23e-4 |
| ProperHAE | 0.1 | 57 | 1.68e-4 |
| ProperHAE | 0.5 | 114 | 7.0e-5 |
| ProperHAE | 1.0 | 114 | 7.8e-5 |
| PortHAE | 0.01 | 52 | 1.84e-4 |
| PortHAE | 0.1 | 48 | 1.89e-4 |
| PortHAE | 0.5 | 77 | 1.32e-4 |
| PortHAE | 1.0 | 77 | 1.38e-4 |

Every physics model hit early stopping within 48–114 epochs, with validation
loss still noisy. CAE trained for 174 epochs. So part of CAE's advantage below
is simply that it trained longer (see Caveats).

HAE's physics term was `0.000000–0.000002` from epoch 10 onward. The variance
penalty was satisfied almost immediately, which is what the VHAE degeneracy in
§2.6 predicts.

### 4.2 OOD reconstruction (`results/reconstruction_metrics.json`)

Evaluated on the held-out mass ranges; lower MSE, higher overlap.

| Model | Test1 MSE | Test1 overlap | Test1 PC1 | Test2 MSE | Test2 overlap | Test2 PC1 |
|---|---:|---:|---:|---:|---:|---:|
| **CAE** | **1.80e-5** | **0.999993** | 0.788 | **1.90e-5** | **0.999992** | 0.825 |
| HAE (λ=0.01†) | 1.32e-4 | 0.999936 | **0.651** | 1.29e-4 | 0.999937 | **0.692** |
| ProperHAE λ=0.01 | 1.08e-4 | 0.999949 | 0.714 | 1.05e-4 | 0.999953 | 0.747 |
| ProperHAE λ=0.1 | 1.61e-4 | 0.999922 | 0.752 | 1.44e-4 | 0.999932 | 0.783 |
| ProperHAE λ=0.5 | 8.09e-5 | 0.999962 | 0.745 | 6.14e-5 | 0.999972 | 0.776 |
| ProperHAE λ=1.0 | 9.05e-5 | 0.999956 | 0.728 | 6.81e-5 | 0.999967 | 0.757 |
| PortHAE λ=0.01 | 1.68e-4 | 0.999925 | 0.627 | 1.56e-4 | 0.999927 | 0.660 |
| PortHAE λ=0.1 | 1.63e-4 | 0.999925 | 0.740 | 1.68e-4 | 0.999927 | 0.770 |
| PortHAE λ=0.5 | 1.38e-4 | 0.999940 | 0.729 | 1.12e-4 | 0.999949 | 0.760 |
| PortHAE λ=1.0 | 1.43e-4 | 0.999933 | 0.718 | 1.36e-4 | 0.999941 | 0.744 |

† Labelled "HAE λ=0.1" in `reconstruction_metrics.json`, but the checkpoint was
trained at λ=0.01 (see Caveats). The λ=0.01 ProperHAE/PortHAE rows come from
`results_not_trained_enough/results/final_metrics.json`.

What the table shows:

1. **CAE reconstructs best**, with 3–9× lower MSE than any physics model. All
   models reach overlap > 0.9999, so on clean OOD signals the differences are
   in the fourth decimal of overlap.
2. **None of the models loses accuracy out of distribution.** Test2 (extreme
   mass ratio, the harder extrapolation) is within about 5 % of Test1 for every
   model, and noticeably *better* for ProperHAE λ ≥ 0.5 (about 25 % lower MSE)
   and PortHAE λ=0.5 (about 19 % lower). The mass-based split therefore does
   not currently stress the models.
3. **Among physics models, ProperHAE λ=0.5 is best** (Test2 MSE 6.1e-5, about
   3× CAE). Within both ProperHAE and PortHAE, λ = 0.5–1.0 beats λ = 0.1, so a
   stronger physics term *helps* reconstruction here instead of competing with
   it. The same pattern shows in validation MSE (§4.1).
4. **PortHAE is consistently worse than ProperHAE** at the same λ (by about
   1.5–2× in MSE at λ ≥ 0.5). The extra dissipation freedom has not yet
   improved accuracy.
5. **PC1 ratio:** every physics model has a lower PC1 than CAE (0.63–0.78 vs
   0.79–0.83), meaning the latent trajectories use more than one direction.
   For ProperHAE and PortHAE, PC1 falls steadily as λ increases from 0.1 to
   1.0. HAE (0.65) and PortHAE λ=0.01 (0.63) have the lowest PC1, but for HAE
   the spread comes from two parallel strands, not an orbit (§4.3).

### 4.3 Latent phase-space portraits (`plots/phase_space_*.png`)

Each plot shows 10 test samples with the first canonical pair (q₀ vs p₀; z₀ vs
z₁ for CAE), coloured by latent time index.

- **CAE:** a horseshoe or arc, consistent across samples, with late-time
  (near-merger) points scattered below it. It has structure but is not a
  closed orbit.
- **HAE:** points lie along a narrow diagonal band (p₀ ≈ −q₀) made of two
  parallel strands. That is close to a 1-D manifold with no rotation, i.e.
  the physics term produced no Hamiltonian-like motion.
- **ProperHAE λ=0.5:** a **closed elongated loop** in every sample. Early and
  mid-time points mix along the loop, so it is traversed repeatedly
  (oscillation), which is what a conservative Hamiltonian system should look
  like. The latent scale is about 10× smaller than for CAE/HAE
  (q₀ ∈ [−0.2, 0.05]).
- **PortHAE λ=0.5:** almost identical to ProperHAE λ=0.5, with the same loop at
  the same scale. The extra dissipation is not visible in the first
  canonical pair.
- In every model, the latest (yellow, near-merger) points fall off the main
  loop. There the chirp frequency rises quickly and one latent step (7.8 ms)
  undersamples the motion, so the finite-difference ż is least reliable.

### 4.4 Learned dissipation in PortHAE

Eigenvalues of `R = L Lᵀ` from the checkpoints (8 × 8, D=4):

| λ | Eigenvalues of R (ascending) |
|---:|---|
| 0.1 | 0, 0, 1.6e-4, 1.5e-3, 2.4e-3, 0.079, 0.274, **2.011** |
| 0.5 | 0, 0, 3e-5, 8.1e-4, 3.2e-3, 0.044, 0.177, **1.001** |
| 1.0 | 0, 0, 1.2e-4, 4.0e-4, 5.7e-3, 0.024, 0.248, **0.332** |

- R has **not** fallen to zero. It has become **effectively low-rank**: two or
  three directions carry all the dissipation and the rest are ≈ 0 (from an
  initial `0.01·I`). The model has learned that energy leaves through a small
  number of directions.
- **The largest eigenvalue shrinks as λ grows** (2.0 → 1.0 → 0.33): a stronger
  physics weight pushes PortHAE toward conservative (ProperHAE-like)
  dynamics. This fits the near-identical phase portraits in §4.3 and
  suggests the finite-difference ż is fitted more easily without
  dissipation.

### 4.5 Denoising real whitened LIGO strain (`results_not_trained_enough/results/report.md`)

Noisy `h1_strain` was fed to the models trained on clean signals and compared
with the clean target. This was run only for HAE and CAE.

| Test set | Model | MSE | Overlap |
|---|---|---:|---:|
| Test1 | HAE | 1.015 | 0.256 |
| Test1 | CAE | 1.038 | 0.248 |
| Test2 | HAE | 1.065 | 0.207 |
| Test2 | CAE | 1.089 | 0.199 |

- **Neither model denoises in a useful way yet.** Because the target is
  normalised to unit variance, MSE ≈ 1 is about what outputting zeros would
  give, and overlap ≈ 0.2–0.26 is low.
- HAE is consistently slightly better than CAE (about 2 % lower MSE and
  +0.008 overlap on both test sets), even though its clean reconstruction is
  7× worse. That is a small hint that the physics prior helps robustness, but
  the effect is far too small to claim without error bars.
- ProperHAE and PortHAE have not been tested on noisy strain.

### 4.6 Parameter recovery from frozen latents (`results/mlp_param_results.json`, run 2026-09-25)

> Full write-up (method, baseline-relative table, discussion, follow-ups):
> [`MLP_PARAM_ESTIMATION.md`](MLP_PARAM_ESTIMATION.md).

`mlp_param_estimator.py` freezes each encoder, flattens its latent trajectory
(448 inputs for D=2, 896 for D=4), and trains an MLP probe
(`in → in/2 → in/4 → 4`) on the training split to predict
`[mass1, mass2, spin1z, spin2z]`. It is scored by RMSE in physical units on
the two OOD test sets.

**Reference: predict the training-set mean for everything.** A useful probe
must beat these numbers:

| | mass1 | mass2 | spin1z | spin2z |
|---|---:|---:|---:|---:|
| Test1 constant-mean RMSE | 12.77 | 16.16 | 0.281 | 0.276 |
| Test2 constant-mean RMSE | 27.73 | 13.69 | 0.293 | 0.273 |

**Probe RMSE** (lower is better; **bold** = best in column):

| Model | Val MSE (norm.) | m1 T1 | m1 T2 | m2 T1 | m2 T2 | s1z T1 | s1z T2 | s2z T1 | s2z T2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAE | 0.527 | **18.57** | **27.99** | 6.62 | 9.11 | 0.250 | 0.229 | 0.297 | 0.321 |
| HAE (λ=0.01†) | 0.506 | 20.88 | 29.96 | 7.50 | 10.43 | 0.225 | 0.206 | **0.290** | 0.326 |
| ProperHAE λ=0.1 | 0.443 | 19.48 | 28.92 | 6.55 | 8.82 | 0.218 | 0.227 | 0.305 | 0.339 |
| ProperHAE λ=0.5 | 0.440 | 19.06 | 28.14 | 5.82 | 8.34 | 0.203 | 0.193 | 0.304 | **0.318** |
| ProperHAE λ=1.0 | 0.436 | 18.91 | 28.64 | 6.56 | 9.36 | 0.175 | 0.205 | 0.308 | 0.335 |
| PortHAE λ=0.1 | 0.454 | 20.26 | 29.46 | **5.31** | **8.03** | 0.204 | 0.202 | 0.298 | 0.327 |
| PortHAE λ=0.5 | 0.456 | 21.02 | 29.87 | 6.33 | 8.89 | **0.161** | **0.174** | 0.328 | 0.357 |
| PortHAE λ=1.0 | **0.412** | 19.99 | 28.67 | 6.40 | 8.80 | 0.168 | 0.188 | 0.311 | 0.325 |

† Labelled "HAE λ=0.1" in the JSON; see Caveat 2.

Findings, parameter by parameter:

- **mass1: no model recovers it.** Every probe is *worse* than predicting the
  training mean, by 45–65 % on Test1 and slightly on Test2. On Test1 the true
  mass1 (50–65) lies inside the training range, but combined with an unusually
  light secondary. A likely explanation is that the latents encode mostly
  chirp mass (which dominates the inspiral phase evolution), and the probe
  turns a low chirp mass into a low mass1 using the correlation it saw in
  training. On Test2 (mass1 > 65) the probe would also have to extrapolate
  beyond any mass1 it saw. CAE comes out least bad.
- **mass2: all models recover it well.** RMSE is 5.3–7.5 against a baseline of
  16.2 on Test1, and 8.0–10.4 against 13.7 on Test2. **PortHAE λ=0.1 is best**
  (−67 % / −41 % vs the baseline), then ProperHAE λ=0.5. Most physics models
  beat CAE, and **HAE is worst**.
- **spin1z: physics models are clearly ahead.** CAE improves on the baseline
  by only 11 % / 22 %. **PortHAE λ=0.5 improves by 43 % / 41 %**, with PortHAE
  λ=1.0 and ProperHAE λ=1.0 close behind. For both ProperHAE and PortHAE,
  λ ≥ 0.5 beats λ = 0.1 on spin1z in almost every case, so a stronger
  physics term puts more spin information into the latent space. This is the clearest benefit of the
  Hamiltonian prior so far.
- **spin2z: no model recovers it.** Every probe is at or worse than the
  baseline, which fits the secondary spin's weak effect on the waveform.
- **Overall (in-distribution val MSE):** all six ProperHAE/PortHAE latents
  beat both CAE and HAE (0.41–0.46 vs 0.51–0.53). PortHAE λ=1.0 is best.

This reverses the reconstruction ranking: **CAE reconstructs best but has the
least parameter-informative latent space**, especially for spin. The
dissipative PortHAE at λ ≥ 0.5 gives the most informative one.

Confounds: ProperHAE and PortHAE use D=4 (twice the latent size of CAE/HAE),
so part of their advantage may be capacity rather than physics. The probe was
trained with a single seed, and the test sets are small (66 and 140 samples),
so differences under about 10 % are within noise.

---

## 5. Caveats on the Current Results

1. **Undertrained.** The checkpoints in `checkpoints/` are byte-identical
   (md5) to the ones in `results_not_trained_enough/checkpoints/`: the
   200-epoch, patience-20 run. The physics models stopped at 48–114 epochs,
   while CAE ran for 174, so the CAE-vs-physics comparison is not yet fair.
2. **Mislabelled HAE checkpoint.** `checkpoints/hae/lambda0.1_best.pt` is the
   same file as `results_not_trained_enough/checkpoints/hae_best.pt`, which the
   training log shows was trained with **λ = 0.01**. The evaluation scripts
   and `reconstruction_metrics.json` label it "HAE λ=0.1".
3. **Incomplete λ sweep.** There is no HAE checkpoint at λ ∈ {0.25, 0.5, 1.0}
   and no ProperHAE/PortHAE checkpoint at λ = 0.25, so `lambda_sweep.py`
   cannot yet produce the sweep described in `training/config.py`.
4. **Learning-rate mismatch.** Every checkpoint used lr = 1e-3. The current
   `PHYSICS_LR_GRID` (5e-5 to 5e-4) does not include that value, so the next
   hparam search will explore a different regime from the one that produced
   these results.
5. **Single seed, no error bars.** Differences under about 10–20 % in MSE
   (for example PortHAE vs ProperHAE at λ=1.0, or HAE vs CAE on noisy strain)
   should be read as unconfirmed.
6. **Not yet run:** `ood_generalization.py` (train-vs-test gap for the physics
   models), `noise_robustness.py` for ProperHAE/PortHAE, and
   `compare_models.py`.

---

## 6. Suggested Next Steps

1. Run the full `train_all.py` (500 epochs, patience 200) so every model gets
   the same budget. Include lr = 1e-3 in `PHYSICS_LR_GRID`, or at least check
   that the lower rates converge within that budget.
2. Retrain HAE at the intended λ values and fix the λ label.
3. Run noise robustness for ProperHAE/PortHAE. The HAE result suggests
   denoising, not clean reconstruction, is where a physics prior could pay
   off.
4. Log diagnostics for the failure modes in §2.6: `Var_t(H)` against the
   spread of H across samples (VHAE), mean `‖∇H‖` (ProperHAE/PortHAE), and
   `eig(R)` over training (PortHAE).
5. Plot all canonical pairs (not only q₀/p₀) to see whether PortHAE's dominant
   dissipation direction shows up in another pair.
6. Separate latent capacity from physics in the parameter probe (§4.6): train a
   D=4 CAE, repeat the probe over several seeds, and add chirp mass and mass
   ratio as probe targets to test the chirp-mass explanation for the mass1
   failure.
