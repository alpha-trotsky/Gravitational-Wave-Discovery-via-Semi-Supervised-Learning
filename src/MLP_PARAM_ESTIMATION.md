# Parameter Recovery from Frozen Latents (MLP Probe)

Companion to [`SOURCE_OVERVIEW.md`](SOURCE_OVERVIEW.md) (code layout) and
[`ARCHITECTURES_AND_RESULTS.md`](ARCHITECTURES_AND_RESULTS.md) (model
descriptions and all other results). This file covers one experiment: how
much physical-parameter information each model's latent space contains.

**Run:** 2026-09-25, CPU, `python src/evaluation/mlp_param_estimator.py`
(default settings). Raw output: `results/mlp_param_results.json`.

---

## TL;DR

- **The physics models' latent spaces carry more recoverable parameter
  information than CAE's**, even though CAE reconstructs waveforms best.
  All six ProperHAE and PortHAE models beat CAE and HAE on in-distribution
  probe loss (0.41–0.46 vs 0.51–0.53).
- **spin1z is the clearest win.** PortHAE λ=0.5 cuts error by 41–43 %
  relative to a constant baseline; CAE cuts it by only 11–22 %.
- **mass2 is recovered well by every model.** PortHAE λ=0.1 is best and HAE
  is worst.
- **mass1 and spin2z are not recovered by any model.** Every probe is worse
  than predicting the training mean.
- **Confound:** ProperHAE and PortHAE have twice the latent size of CAE and
  HAE (D=4 vs D=2), so part of their advantage may be capacity, not physics.

---

## 1. Method

For each available checkpoint:

1. **Freeze the encoder** and extract the latent trajectory `z ∈ R^(2D × 112)`
   for every sample in train (no jitter), val, test1 and test2. For the
   physics models, `z = cat(q, p)`.
2. **Flatten** it: 448 inputs for D=2 (CAE, HAE), 896 for D=4 (ProperHAE,
   PortHAE).
3. **Train an MLP probe** to predict `[mass1, mass2, spin1z, spin2z]`, with
   targets z-scored using training-set mean and std:
   ```
   ParamMLP: in → in/2 → ReLU → in/4 → ReLU → 4
   ```
   Adam at lr = 1e-3, batch size 64, up to 300 epochs, early stopping on val
   MSE with patience 30. The best val state is restored.
4. **Evaluate** RMSE in physical units (solar masses, dimensionless spin) on
   the two OOD test sets.

Data splits (from `core/data.make_splits`):

| Split | N | Selection |
|---|---:|---|
| train | 1017 | `m1 ≤ 65 & m2 ≤ 65`, excluding the test1 region |
| val | 179 | 15 % random holdout from the train range |
| test1 | 66 | `m2 ∈ [5,15] & m1 ∈ [50,65]`: asymmetric, intermediate mass ratio |
| test2 | 140 | `m2 ≤ 20 & m1 > 65`: extreme mass ratio, heavy primary |

(The dataset's minimum mass2 is 10.0, so test1 effectively has m2 ∈ [10, 15].)

Checkpoints probed: CAE, HAE (one checkpoint), and ProperHAE and PortHAE at
λ ∈ {0.1, 0.5, 1.0}. The HAE λ ∈ {0.25, 0.5, 1.0} and all λ = 0.25
checkpoints do not exist yet and were skipped.

### Fix applied to the script

The first run crashed in `_load_labels` with
`TypeError: Indexing elements must be in increasing order`. h5py fancy
indexing needs sorted indices, but `val_idx` is a random pick (`rng.choice`)
and must stay in loader order so the labels line up with the latents. The fix
reads each full label column and indexes it in numpy:

```python
cols = [f[f"injection_parameters/{k}"][:][indices].astype(np.float32)
        for k in PARAM_KEYS]
```

Separately, printing `λ` fails on the Windows cp1252 console when output is
piped. Run with `PYTHONIOENCODING=utf-8` if you hit that.

---

## 2. Reference: constant-prediction baseline

The script reports only RMSE. To judge whether a probe has learned anything,
it has to be compared with the error from always predicting the
**training-set mean** (mass1 45.65, mass2 28.56, spin1z 0.493, spin2z 0.501):

| | mass1 | mass2 | spin1z | spin2z |
|---|---:|---:|---:|---:|
| Test1 constant-mean RMSE | 12.77 | 16.16 | 0.281 | 0.276 |
| Test2 constant-mean RMSE | 27.73 | 13.69 | 0.293 | 0.273 |
| *Test1 own std (spread within the set)* | *4.48* | *1.42* | *0.279* | *0.275* |
| *Test2 own std* | *4.14* | *2.85* | *0.293* | *0.271* |

The masses in each test set are tightly clustered (std 1.4–4.5), but far from
the training mean. So most of the baseline error is an *offset*: a probe can
beat the baseline by getting the average mass of the test region right,
without resolving differences between individual samples.

---

## 3. Results

### 3.1 Probe RMSE (lower is better; **bold** = best in column)

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

† `results/mlp_param_results.json` labels this "HAE λ=0.1", but the
checkpoint was trained with λ = 0.01 (see
`ARCHITECTURES_AND_RESULTS.md`, Caveat 2).

"Val MSE (norm.)" is the probe's best validation loss averaged over the four
z-scored targets. It is in-distribution, and 1.0 would mean no information.

### 3.2 Change vs constant baseline (negative = better than baseline)

| Model | m1 T1 | m1 T2 | m2 T1 | m2 T2 | s1z T1 | s1z T2 | s2z T1 | s2z T2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAE | +45 % | +1 % | −59 % | −33 % | −11 % | −22 % | +7 % | +18 % |
| HAE (λ=0.01†) | +64 % | +8 % | −54 % | −24 % | −20 % | −30 % | +5 % | +19 % |
| ProperHAE λ=0.1 | +53 % | +4 % | −59 % | −36 % | −22 % | −22 % | +10 % | +24 % |
| ProperHAE λ=0.5 | +49 % | +1 % | −64 % | −39 % | −28 % | −34 % | +10 % | +16 % |
| ProperHAE λ=1.0 | +48 % | +3 % | −59 % | −32 % | −38 % | −30 % | +12 % | +23 % |
| PortHAE λ=0.1 | +59 % | +6 % | **−67 %** | **−41 %** | −27 % | −31 % | +8 % | +20 % |
| PortHAE λ=0.5 | +65 % | +8 % | −61 % | −35 % | **−43 %** | **−41 %** | +19 % | +31 % |
| PortHAE λ=1.0 | +57 % | +3 % | −60 % | −36 % | −40 % | −36 % | +13 % | +19 % |

---

## 4. Findings

### mass2: recovered by every model; physics models lead
All probes land 24–67 % below the baseline. **PortHAE λ=0.1 is best** on both
test sets, followed by ProperHAE λ=0.5. Every physics model beats CAE on
Test1, and all but ProperHAE λ=1.0 beat it on Test2. **HAE is worst**, which
fits its collapsed, nearly one-dimensional latent (see
`ARCHITECTURES_AND_RESULTS.md` §4.3).

Given the offset point in §2, much of this gain may be the probe placing the
test sets in the right low-mass2 region, not resolving mass2 differences
within them.

### spin1z: the clearest benefit of the physics prior
CAE barely improves on the baseline (−11 % / −22 %). The physics models do
much better, and the dissipative model does best: **PortHAE λ=0.5 reaches
−43 % / −41 %**, with PortHAE λ=1.0 (−40 % / −36 %) and ProperHAE λ=1.0
(−38 % / −30 %) close behind. For both ProperHAE and PortHAE, λ ≥ 0.5 beats
λ = 0.1 in almost every case, so a stronger physics term puts more spin
information into the latent space.

Spin enters the waveform mainly through the phase evolution: spin-orbit
coupling changes how fast the orbit decays. That is exactly what the
ProperHAE/PortHAE losses constrain, since they tie the latent *velocity* to a
learned energy function. The PortHAE result in particular fits its learned
dissipation matrix R standing in for the radiation-reaction rate, which
spin modifies.

### mass1: not recovered by any model
Every probe is **worse than predicting the training mean**: by 45–65 % on
Test1 and by 1–8 % on Test2. CAE is the least bad.

- On **Test1**, mass1 (50–65) is inside the training range, but paired with an
  unusually light secondary (10–15). The likely explanation is that the
  latents mainly encode **chirp mass** `M_c = (m1 m2)^(3/5) / (m1+m2)^(1/5)`,
  which controls the inspiral phase evolution, and only weakly encode the
  mass ratio. A light secondary gives a low chirp mass, and the probe then
  predicts a low mass1 using the m1–M_c correlation it saw in training.
- On **Test2**, mass1 > 65 lies entirely outside the training range, so the
  probe would have to extrapolate. MLPs with ReLU rarely do that well.

### spin2z: not recovered by any model
All probes are 5–31 % worse than the baseline. That is expected: the
secondary's spin has a much weaker effect on the waveform, especially at
these mass ratios where the secondary is light.

### Overall
- **In-distribution probe loss:** all ProperHAE and PortHAE latents
  (0.41–0.46) beat both CAE (0.53) and HAE (0.51). PortHAE λ=1.0 is best.
- **The ranking is the opposite of reconstruction.** CAE has the lowest
  reconstruction MSE, but its latent space holds the least recoverable spin
  information. The physics losses give up some reconstruction accuracy in
  exchange for a more physically meaningful latent space, which is what this
  project was designed to show.

---

## 5. Caveats

1. **Latent capacity confound.** ProperHAE and PortHAE use D=4 (896 probe
   inputs), CAE and HAE use D=2 (448 inputs). Part of the physics models'
   advantage may simply come from a larger latent. A D=4 CAE is needed to
   separate the two effects.
2. **Single seed.** Both the probe and the encoders were trained once. With
   test sets of 66 and 140 samples, differences under about 10 % (for example
   ProperHAE λ=0.5 vs PortHAE λ=1.0 on mass2) are within noise.
3. **Undertrained encoders.** All checkpoints come from the 2026-06-24 run
   (max 200 epochs, patience 20). The physics models stopped at 48–114 epochs,
   while CAE ran for 174.
4. **HAE label.** The only HAE checkpoint was trained with λ = 0.01, not the
   λ = 0.1 its filename and the JSON label suggest.
5. **Offset-dominated baseline.** Because each test set's masses are clustered
   far from the training mean, beating the constant baseline on mass does not
   show that the probe resolves differences *within* a test set. Per-sample
   correlation (R² within each test set) would be the stricter test.

---

## 6. Suggested Follow-ups

1. **Train a D=4 CAE** and probe it, to separate latent capacity from the
   physics prior.
2. **Run several seeds** (probe and, ideally, encoder) and report mean ± std.
3. **Add chirp mass and mass ratio** as probe targets. If chirp mass is
   recovered well while mass1 is not, that confirms the explanation in §4.
4. **Report within-test-set R²** alongside RMSE (Caveat 5).
5. **Probe structured summaries of the latent,** such as the learned H along
   the trajectory, the orbit's period and amplitude, or R's dominant
   eigen-direction for PortHAE, instead of the flattened latent. This would
   show *where* the parameter information lives.
6. Repeat everything once the full `train_all.py` run (500 epochs,
   patience 200, complete λ sweep) is finished.
