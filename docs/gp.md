# Gaussian processes

The multi-fidelity GP over configurable levels and the binomial count GP.

## Multi-fidelity GP: configurable levels

The MFGP fits an ordered list of levels, lowest fidelity first; the last level is the
target. Each level is one (X, Y) set of prepared voxels:

- `model_mean`: mean predicted p per voxel of an event model. With `inference:` (a file
  from `python -m core.surrogates predict`) the saved values are used after checking that
  voxels and normalization match the data; otherwise the event model predicts at fit time
  (context models split each voxel's labeled events with `n_context`).
- `observed_fraction`: simulated hit fraction m/N over all events of each voxel.

`max_voxels` takes a seeded voxel subset (exact GPs scale as O(n³)); levels on the same
partition/fidelity share it. `evaluation` lists held-out partitions whose observed
fraction is scored against one level (default: the top level). Log mode smooths observed
fractions to (m + a)/(N + 2a) and floors model means at `cnp_log_floor`.

```yaml
levels:
  - {source: model_mean, partition: gp_unlabeled, inference: RUN/inference/best/gp_unlabeled_lf.npz}
  - {source: observed_fraction, partition: gp_train, fidelity: hf}
evaluation:
  - {partition: gp_validation, fidelity: hf}
kernel: rbf
n_restarts: 5
output_transform: log
```

Two ways to run it, with the same code and outputs:

- **Combined:** a `mfgp:` block in a training config fits the GP after training with the
  checkpoint `mfgp_checkpoint`; outputs go to `RUN/mfgp/`.
- **Standalone:** `python -m core.surrogates mfgp CONFIG.yaml` with the block above plus
  `data_directory`, `output_directory`, optional `data_config` and `checkpoint` (needed
  only for `model_mean` levels without an inference file). Paths are relative to the YAML.

Every `mfgp:` block lists its levels explicitly. Observed fractions use all events of a
voxel; model means of context models cover the events left after the context split.

Outputs: `training_arrays.npz` (`X_k`, `Y_k`, `events_k`, `index_k`), `fit_arrays.npz`,
`transformed_training_arrays.npz`, `levels.json` (settings, voxel counts, provenance),
`model.pkl`/`model.json`, `evaluation_<e>.npz` + `metrics.json` (one row per evaluation
set), `development_map.npz`, and plots (`training_inputs`, `evaluation_<e>_means`,
`evaluation_<e>_coverage`, `development_map`).

In Python, `core.mfgp_pipeline` builds levels from batches (`BatchLevel`,
`levels_from_batches`, `synthetic_levels` for pseudo data) and `fit_mfgp(X_list, Y_list)`
fits them; the active-learning loop takes the same lists plus `observed_levels` (which
levels a new simulation extends). GP fitting needs GPy/Emukit; event models don't. Classifier uncertainty
is not substituted for the GP posterior uncertainty.

Use the component API or the common runner for training and evaluation. Shared configuration and
metrics improve reproducibility but do not make context-conditioned CNP and
context-free models identical comparisons, or eliminate validation-selection bias.

## GP stage after training

The optional `mfgp` run-config section enables this second stage (levels: see "Multi-fidelity
GP: configurable levels"). It requires the GP dependencies. Large datasets: a level's
`max_voxels` fits the exact GP on a seeded subset of its partition (`index_k` in
`training_arrays.npz`; one stream per partition/fidelity in order of appearance). Evaluation
sets and the development map use every voxel, and event-model training is not subsampled.
Omitting the section preserves neural-only runs. With `lf_validation: true` (the API default), the CNP is selected by LF
validation. With `lf_validation: false` the run uses the final checkpoint after all
configured steps, without selecting on HF. LF validation files and diagnostics are skipped; LF training PR and training
loss history remain available. MFGP hyperparameters use training arrays only.
HF validation is used for reporting; no test files are loaded.

Alongside the CNP artifacts, `mfgp/` contains:

- `model.pkl`, loadable with `core.surrogate_mfgp.load_mfgp`.
- `model.json`: fitted parameters, input hashes, configuration, and conventions.
- `training_arrays.npz`: exact three-level GP training inputs and outputs.
- `metrics.json` and `{lf,hf}_validation.npz`: predictions on real target fractions.
- `{lf,hf}_means.png` and `{lf,hf}_coverage.png`.
- `development_map.npz`: highest-fidelity predictions at available development
  coordinates, with normalized and physical coordinates; a spatial plot when 3D.
- `normalization.json`: original preprocessing needed for new spatial queries.

The existing GP learns constant Gaussian observation noise separately per level.
Reported variance includes that observation noise; CNP scales are not used as
GP noise. LF coverage compares level 0 predictions to LF raw fractions as a
diagnostic even though level 0 was fitted to denoised CNP means. HF coverage
compares the highest level to raw HF validation fractions. Gaussian predictions
are saved without clipping to [0,1]. This is an offline pipeline, not an active
simulation-launch loop, and the development-coordinate map is not a dense grid.

For physical Cartesian queries, normalize with the saved theta offset and scale
before calling `gp.predict(theta_normalized, fidelity=2)`. Predictions remain in
detection-fraction units. Keep normalization alongside the GP checkpoint.

## Log-space MFGP

`mfgp.output_transform: log` fits all three GP levels to natural logarithms;
`identity` remains the shared API default and reproduces linear-space behavior.
Observed-fraction levels use `(m+a)/(N+2a)` with `raw_pseudocount: 0.5` and each voxel's
event count. This Beta(1/2,1/2) posterior-mean
smoothing applies to all HF raw training fractions, including zeros. CNP means
are not counts: they only receive the numerical `cnp_log_floor: 1e-12`.

The log of a smoothed fraction does not average to log p: the pseudocount pushes it up
(≈ a/(Np)), the log of a noisy value pulls it down (Jensen), ≈ +16 % and −12 % at 5 hits,
+78 % and −18 % at 1 hit. Log mode therefore corrects every observed-fraction target by the
exact binomial offset E[log((m+a)/(N+2a))] − log p (`core.mfgp_pipeline.log_target_offset`)
at the rate p of the previous fit: first fit with uncorrected targets, then refits from the
previous hyperparameters (one optimizer run each) until no offset changes by more than 0.01,
at most `max_refits` (default 8; 0 = off). `fit_arrays.npz` holds the offsets
(`log_offset_k`), `levels.json` their mean, `model.json` `refits` (refits, converged,
max_change, bias_correction, point_noise). The synthetic sweep (`scripts/mfgp_noise_sweep.py`) converges in
2–3 refits at 5 hits per point and in 7–8 at 1 hit.

Observation noise: `observation_noise: learned` (default) gives each level one constant
noise variance. With `binomial`, every observed-fraction point also gets its count noise at
the rate of the previous fit, as a fixed variance on top of the learned noise of its level
(`LevelPlusPointNoise`): Var[log((m+a)/(N+2a))] in log mode, p(1 − p)/N in identity mode
(`core.mfgp_pipeline.count_noise`). It is updated in the same refits (`max_refits`; identity
mode refits only for it, stopping once no noise standard deviation changes by more than 1 %);
the first fit has the learned noise only, so a point's noise never depends on its own
observation. Held-out points scored against an observed-fraction level get the same count
noise from their N and the predicted rate in the intervals and scores, never in the mean;
`development_map.npz` has the learned noise only (no N). Per-point noise is stored as
`point_noise_k` in `fit_arrays.npz`. A point whose predicted rate is ≤ 0 or ≥ 1 (possible in
identity mode for dim points) gets no binomial noise and, in log mode, no offset: only the
learned level noise, never a clipped rate. The count is recorded per level in `levels.json`
and `model.json` `refits` (`rates_outside_unit_interval`, last refit) and per evaluation set
in `metrics.json`. Constant noise covers dim and
bright points unequally (log mode: dim too narrow; identity: bright too narrow); on the
synthetic sweep at 50 hits, 1σ/2σ coverage of dim/bright points is 0.59/0.87 and 0.86/0.99
(log) and 0.86/0.99 and 0.56/0.89 (identity) with `learned`, 0.69/0.94 and 0.69/0.95 (log)
and 0.70/0.96 and 0.64/0.94 (identity) with `binomial`, with unchanged bias.

`training_arrays.npz` preserves original values; `fit_arrays.npz` records the
adjusted original-scale targets and offsets, `transformed_training_arrays.npz` records
the exact GP targets (log of the smoothed value minus the offset), and `levels.json` records per model-mean level how many values hit
the floor (`log_floored`).
`MultiFidelityGP.predict` returns the rate mean exp(μ + v_f/2), with v_f the latent GP
variance only, and the lognormal variance of an observation (noise included). The learned
noise stands for the count scatter of m/N, which averages to the rate (E[m/N] = p), so it
widens the variance and intervals but does not raise the mean; before 2026-11-09 the mean
was exp(μ + v/2) with the noise in v, ≈ +9 % per 0.18 log-units² of noise, so log-mode
means of older runs are not comparable. `predict_transformed` returns Gaussian log-space
moments (`include_noise=False`: latent only). `predict_interval` returns asymmetric
equal-tail bounds of the noisy predictive. Save/load preserves these semantics; older
checkpoints default to identity mode.

Validation errors use unchanged raw fractions. Coverage plots use transformed
interval bounds, not mean ± original-scale standard deviation. Positive bounds
cannot cover a zero observed fraction: this diagnostic is not a binomial
posterior-predictive count interval. Lognormal predictions can exceed one.
Observation noise is learned in log space and is not CNP uncertainty.
Active-learning acquisition/refitting currently requires identity mode; the
offline training and prediction workflow supports both transforms.

## Default GP validation metrics

MFGP runs write original-scale, equally weighted per-voxel scores to `metrics.json`: mean bias (predicted minus
observed), MAE, RMSE, Pearson/Spearman correlations, descriptive calibration
intercept/slope, CRPS, and weighted interval score. `interval_metrics` contains
nominal probability, inside/total counts, measured coverage, mean width and
interval score at each 1/2/3-sigma-equivalent level. WIS uses the predictive
median, not the mean. Undefined correlations are JSON null. Existing coverage
keys remain compatible. The low-level MFGP coverage evaluator returns these
scores under `metrics` as well.

The MFGP NLL is a Gaussian/lognormal predictive density in response units; it must not
be ranked against a count-probability NLL. A lognormal model assigns zero
probability to a zero observation; its nonfinite NLL is saved as null with an
explanation rather than silently clipping the observation. Predictive sampling
uses a fixed seed, and the method/draw count are recorded.

These metrics evaluate held-out observations including counting noise. They do
not establish true latent-rate error or tank-wide population accuracy. Spatial
weighting, regional scores and distance-based validation require a specified
physical evaluation design; no volume weighting is inferred from sparse points.

## Binomial count GP

`python -m core.surrogates binomial CONFIG.yaml` fits one spatial GP directly to hit counts:
m ~ Binomial(N, sigmoid(f(θ))), Matérn-5/2 (or RBF) plus a constant kernel, Laplace
posterior (`core/binomial_gp.py`). Several count sets (e.g. LF and HF partitions with the same
physics) share the latent probability; their different event counts N enter the likelihood
exactly, so the counting noise is per voxel by construction. Evaluation draws joint latent
probabilities and then binomial counts (observation-predictive); the NLL is a count probability
mass and must not be ranked against the MFGP's density NLL. Outputs: `observations.json`,
`training_arrays.npz`, `model.pkl`/`model.json`, `evaluation_<e>.npz`, `metrics.json`,
`development_map.npz` and coverage / z-score plots. Settings: [docs/configs.md](configs.md).
