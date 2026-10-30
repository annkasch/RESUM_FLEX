# Interchangeable event surrogates

`core.surrogates` is the model-independent API for the single-logit CNP, MLP,
feature-token transformer and histogram boosted decision trees. All consume
`StandardBatch` and produce `EventPrediction`: event logits, probabilities, and
voxel means. There is no Gaussian uncertainty output from these classifiers.

## Python API

```python
from core.surrogates import build_surrogate, Episode, evaluate_surrogate, load_surrogate
from schemas.surrogates import NeuralTraining

# Only the model spec changes between the neural architectures.
model = build_surrogate({"kind": "cnp"}, dim_theta=3, dim_phi=6, seed=0)
training = NeuralTraining(
    sampling={"strategy": "positive_quota", "positive_fraction": 0.05},
    weighting={"strategy": "sampling_correction"},
)
report = model.fit(
    train_batch, training,
    validation=Episode(validation_context, validation_targets),
    selection="voxel_rate_mae",
    checkpoints="outputs/my_run/checkpoints",
)
prediction = model.predict(validation_targets, context=validation_context)
metrics, arrays = evaluate_surrogate(model, validation_targets, context=validation_context)
restored = load_surrogate("outputs/my_run/checkpoints/best")
```

Construct `kind="mlp"`, `kind="transformer"` or `kind="bdt"` using the same factory.
BDT uses `TreeTraining()` and requires the optional `.[bdt]` dependencies.
The CNP requires context observations from the same voxels. Other models ignore
context, which allows fixed evaluation episodes to be shared. They do not gain
access to context labels. All prediction shapes are `[voxel, event]`; the mean
is an unweighted average over exactly those events.

Architecture options live inside the model spec: `encoder` for CNP and
`architecture` for MLP/transformer/BDT. MLP architecture exposes only hidden
widths and dropout, without unused CNP latent settings. Transformer defaults are the revised
122,113-parameter FT-style configuration. Neural initialization seed is explicit
in the factory; training seed controls episode sampling and dropout. BDT's seed
is in its architecture config.

`fit` returns history, the selected step/tree count, and a sampling audit. It
restores the selected model. The optional checkpoint directory contains both
`best/` and `final/`. Without validation, the last step is selected. Each neural
fit creates a fresh optimizer and continues from the module's current weights;
each BDT fit resets the estimator. This is not optimizer-state resume support.

## One experiment entry point

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 1. Train an event model (data preparation runs automatically from data_config)
.venv/bin/python -m core.surrogates train TRAINING.yaml
# 2. Mean predicted p per voxel of a partition (context-free models; labels not needed)
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION
# 3. Fit the multi-fidelity GP on its configured levels
.venv/bin/python -m core.surrogates mfgp GP.yaml
```

Data come from a dataset provider (see "Datasets and providers").
Change model and training settings using the API
options below; BDT fits natural events. `selection` picks the checkpoint metric
(`bernoulli_log_loss` for natural-event runs, `average_precision` for mixup ablations).
Change `output_directory` for every new run: existing nonempty runs are never
overwritten. Data/output paths resolve relative to the YAML file.

The runner loads prepared LF training and LF/HF validation batches. Inputs must
already have the training-only normalization applied. It never reads test data.
Only LF validation selects a model; HF is scored after selection. Training PR is
in-sample. Files under the output directory are uniform across model families:

- `experiment.json`: resolved configuration, input-file hashes, normalization,
  feature metadata and split manifest when available.
- `checkpoints/{best,final}/`: versioned model manifest and backend weights.
- `history.json`, `sampling_audit.json`, `metrics.json`.
- `{lf,hf}_{train,validation}_{best,final}.npz` where evaluated: logits, labels,
  voxel means, residuals, and PR arrays. Only LF training is evaluated.
- Shared PNG/PDF validation-history, mean/residual and PR plots.

The Python entry point is `core.surrogates.experiment.run_experiment(config)`.
Notebooks can call it and display artifacts, without duplicating a training loop.
The CLI limits Torch threads to one; BDT fitting limits its thread pools to one.
CUDA is opt-in through `training.device`; default examples run on CPU.

## Blinded test sets

Test partitions are blinded by default. `test_fidelities` declares which test sets exist,
but a training run evaluates them only with `unblind_test: true`; otherwise it writes
`test_evaluation.json` with `status: blinded` and never reads the test data. The MFGP
does the same: evaluation on `test`, `cnp_test` or `gp_test` needs `unblind_test: true`,
and test partitions can never be GP training levels. Make all model and GP choices on
validation, then unblind once.

## Validation metrics

The event models output per-event hit probabilities, so they are judged as
probabilities: right order (discrimination) and right level (calibration).
`evaluate_surrogate` reports, for one episode (targets only, padding excluded):

| Metric | Level | Meaning | Ideal |
|---|---|---|---|
| `bernoulli_log_loss` | event | proper score; recommended `selection` for natural-event runs | low |
| `log_loss_skill` | event | `1 - LL / LL_const`, constant = training prevalence | > 0 |
| `brier_score` | event | mean `(y - p)^2` | low |
| `average_precision`, `lift` | event | order of events only; lift = AP / prevalence | lift > 1 |
| `calibration_z` | all | `(Σk - Σp) / sqrt(Σp(1-p))`, overall level | \|Z\| < 3 |
| `calibration_chi2_per_bin` | event bins | 10 equal-count bins of p, mean `(K-M)^2/V` | ≈ 1 |
| `voxel_z_mean`, `voxel_z2_mean` | voxel | `z = (k - Σp) / sqrt(Σp(1-p))` | 0, 1 |
| `voxel_rate_mse_denoised` | voxel | mean of `(o - q)^2 - o(1-o)/(n-1)` | ≈ 0 |
| `spearman_r` | voxel | rank correlation of `o` and `q` (trend) | near ceiling |

Per voxel: `o` = hits / target events, `q` = mean predicted p over the same
targets, `n` = target events. The subtracted term is the unbiased binomial noise
estimate, so `voxel_rate_mse_denoised` estimates the real `(t - q)^2`; it can be
slightly negative and is never clipped. `voxel_rate_mae` is kept for old runs
only: with few hits per voxel it is dominated by noise and favours the median
count (it prefers 0 in voxels with < 0.7 expected hits). AP needs natural
probabilities only for its baseline; use it for mixup ablations, whose outputs are
scores, not probabilities. The log-loss skill needs the training prevalence and is
`None` without it.

Evaluation layout: each voxel's events are permuted with `validation_seed`. The
targets are the events after `validation_context_pool = max(validation_context_events,
*validation_context_sweep)`; the context is the first `n` of the pool. Every context
size therefore scores the same targets, and contexts are nested.

Read-out after training (held-out splits only, not used for selection):

- `spearman_noise_ceiling`: median Spearman r over 200 label sets redrawn from the
  model's own p, i.e. what a model equal to the truth would reach at this noise.
- `*_ci95`: 95% bootstrap intervals over voxels (200 resamples) for log loss, AP and
  Spearman r. Use them to compare runs.
- `context_sweep.json` / `context_sweep.png` (context models only): the metrics at
  every size in `validation_context_events` + `validation_context_sweep`, plus the
  standard size with all context labels set to 0. A flat curve, or zeroed labels
  scoring like observed ones, means the model ignores its context labels.
- `{fidelity}_calibration.png`: observed hit fraction vs mean p per bin, Wilson 95%.

Seeds: `SeedSequence([validation_seed, k])`, k = 0 selection subset, 1 noise
ceiling, 2 bootstrap.

## Loss, sampling and weights

These settings are deliberately separate:

- `focal_gamma: 0` is stable BCE on logits; positive gamma enables focal BCE.
- Natural sampling has no quota. It can use no weights or explicit class weights.
- `positive_quota` requires `sampling_correction`, using the existing per-voxel
  sampler and actual integer quota. No extra class rebalancing is added.
- BDT fits all original events, optionally with explicit class weights. It rejects
  neural losses, oversampling settings and sampling correction.

The API rejects unknown keys, including legacy top-level mixup_alpha and
Gaussian-loss settings. Class-aware mixup uses the nested sampling config below.
There is no silent fallback from one loss or sampler to another. Class weights
and focal losses can change probability calibration; the saved config records
these choices. AP is explicitly undefined (`None`, with empty PR arrays) for
validation sets with no positives; AP selection then fails before fitting.

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

## Class-aware mixup

`core.mixup.ClassAwareMixupSource` samples voxels uniformly with replacement.
For each voxel occurrence in each batch, it freshly partitions the full source
set into disjoint context and target pools, independently of labels, in the
requested context/target proportion. Both parents of a mixture come from the
same pool. Positive partners can repeat within their pool; weights are independent
per pair. A pool with only one class returns real events. A lone positive may
serve either side across batches; it is not permanently reserved for targets.

`mix_context=False` samples real contexts; `True` mixes context too. For equal
seeds, both modes use identical splits and targets. Real-event and negative-anchor
sampling uses replacement only if a pool is too small for the requested batch.
This is batch sampling, without a once-per-epoch negative coverage guarantee.
CNP, MLP, and transformer training support this sampler through the shared API:

```yaml
training:
  backend: neural
  sampling:
    strategy: class_aware_mixup
    mixup:
      alpha: 0.2
      mix_context: false
  weighting:
    strategy: none
```

Run `python -m core.surrogates train CONFIG.yaml` with these settings.
Change the model kind for MLP or transformer. Only CNP uses context observations;
other neural models use the same mixed targets. The context size follows `training.context`
(see "Episode building" below). Mixup defaults to real context.
Stable BCE (`focal_gamma: 0`) accepts soft target labels. BDT does not support
this sampler. Sampling correction and class weights are rejected with mixup;
no probability-bias correction is implied. Positive quotas cannot be combined
with it. Permutation mixup remains a standalone utility.

### Episode building

Each optimizer step draws `batch_size` voxels (with replacement) and `n_events`
events from each voxel. These events are split into context and targets:

```yaml
training:
  batch_size: 16        # voxels per step
  n_events: 750         # events per voxel per step
  targets: disjoint     # or joint
  context:
    size: random        # one size per step, uniform in [n_min, cap], shared by all voxels
    n_min: 16
    max_fraction: 0.5   # cap = floor(0.5 * n_events) = 375; null = no cap
# fixed size instead:
#  context: {size: fixed, n: 250}
```

- `targets: disjoint` (default) scores only the non-context events. `joint` also
  scores the context events, as in Garnelo et al. (2018); it is limited to
  natural sampling. With the same seed both modes draw the same events. Joint
  puts loss weight on labels the model has already seen in its context, so it
  can over-trust the context; evaluation is always disjoint.
- Without a cap, the context may use up to `n_events - 2` (disjoint) or
  `n_events - 1` (joint) events.
- Old configs with `n_context_min` / `n_context_max` still load. They become a
  random context with an absolute cap `context.n_max`, so their range does not
  change with `n_events` (e.g. in fine-tuning).
- Run length: `n_steps`, or `n_passes` (not both; default 10000 steps). One pass
  means as many scored target events as there are training events. The steps are
  `ceil(n_passes * training_events / targets_per_step)`, with
  `targets_per_step = batch_size * (n_events - mean context size)` for disjoint and
  `batch_size * n_events` for joint targets; the mean context size of a random
  context is `(n_min + cap) / 2`. Example: 1999 voxels x 1500 events, 16 x 750,
  context 16-375, `n_passes: 10` -> 3380 steps.
- The sampling audit records `targets`, the resolved `context_range`, `n_steps`,
  `requested_passes` and the realized `target_passes`.

Validation and reported train/validation metrics use real binary events. The
sampling audit records soft-label mass, mean target label, and soft-label event
count instead of treating soft labels as positive-event counts. Resolved mixup
settings are saved in checkpoints and experiment metadata.

## Optional real-plus-mixup BCE

For CNP, MLP, or transformer, add this to class-aware mixup training:

```yaml
training:
  sampling:
    strategy: class_aware_mixup
    mixup:
      alpha: 0.2
      mix_context: false
  weighting:
    strategy: none
  focal_gamma: 0
  objective:
    strategy: real_plus_mixup
    mixup_loss_weight: 0.001
    real_target_ratio: 1.0
```

The objective is `mean(real BCE) + mixup_loss_weight * mean(mixed BCE)`.
Both branches use the same context observations and model parameters, with one
optimizer update. Real targets are sampled uniformly from that batch's target
source pool, independently of labels. No source event crosses the context/target
split. Positive source events can be reused as mixture parents. Real targets
are sampled without replacement unless the requested count exceeds the pool.

`real_target_ratio` adds `max(1, round(ratio * mixed_target_count))` real targets
per voxel; it does not reduce the mixed target count. A ratio of 1 doubles target
predictions. Each branch is averaged separately, so its sample count does not
implicitly multiply its loss weight. Zero mixup loss weight is allowed. The
Beta `alpha` and the loss coefficient have distinct meanings. Extra class weights
and focal loss are rejected for this objective. `objective.strategy: single`
remains the default and uses the existing one-branch objective.

The sampler's optional `next(real_target_ratio=...)` returns
`(context, mixed_targets, real_targets)`; without the argument its two-result
interface and mixed-batch random sequence are unchanged. Shared training handles
this interface automatically.

History records `training_real_bce`, `training_mixup_bce`, and their weighted sum
as `training_loss`, averaged over each reporting interval. Checkpoints record
objective settings and sampling counts; the audit includes both target counts,
real positive counts and total target predictions. Validation still uses real
binary events. This is an experimental augmentation objective, not an unbiased
probability correction or an established performance improvement.

To try this objective, set `training.objective` in a training config as above.

## Legacy two-output CNP

Prefer the single-logit `cnp`. Under
`theory-truth` the two outputs only enter through one effective logit
`mu_logit / d(sigma)`, so the second channel adds no representable predictions
and is not identified by binary labels. `legacy_cnp` is kept to reproduce the
original RESuM objective and older checkpoints.

`model.kind: legacy_cnp` wraps the existing `build_cnp` architecture without
changing its decoder or likelihood definitions. Set `training.loss` explicitly:

- `theory-truth`: Bernoulli NLL on the effective logit computed from both outputs.
- `practice-truth`: the original Gaussian NLL on the transformed mean and scale.

Single-logit models keep `training.loss: bernoulli` (the default). Mismatched
model/loss settings fail validation. Legacy losses require `focal_gamma: 0`.
Natural sampling (optionally with explicit class weights), class-aware mixup
alone, `objective.strategy: real_plus_mixup`, and real-target oversampling are
supported. For oversampling, set

```yaml
sampling: {strategy: positive_quota, positive_fraction: 0.05}
weighting: {strategy: sampling_correction}
```

Each event's legacy loss (theory-truth BCE or practice-truth Gaussian NLL) is
multiplied by its per-voxel importance weight p/q or (1-p)/(1-q) before the
event mean, so the expected loss matches uniform real targets. For theory-truth
this is an unbiased estimate of the natural Bernoulli risk. Practice-truth is not
a Bernoulli likelihood, so the weights only restore its natural-sampling objective. For the combined objective,
**both** branches use the selected legacy loss and one optimizer update:
`mean(real legacy loss) + mixup_loss_weight * mean(mixed legacy loss)`.

The `practice-truth` combined history uses `training_real_nll` and
`training_mixup_nll`; `theory-truth` retains the BCE component names. These replace
the BCE-only restriction described above specifically for the legacy adapter.
Both output channels receive gradients. The Beta mixing parameter and relative
loss weight remain separate settings.

Shared predictions use `resum_binary_logits` (whose sigmoid is the transformed
mean), not the historical `predict_beta` helper's sigmoid of the raw mean logit.
The transformed scale is preserved as `EventPrediction.legacy_scale` and in
saved evaluation arrays under `legacy_scale`. It is the original scale proxy,
not a calibrated uncertainty estimate or an MFGP noise variance. Shared metrics,
PR curves, mean plots and MFGP preparation use the effective probabilities.
Validation labels stay real and binary even with the Gaussian training loss.
The common `bernoulli_log_loss` metric remains Bernoulli log loss for comparison;
it is not the Gaussian training objective.

Select the combined objective explicitly using the settings above. New checkpoints
roundtrip both output channels through the standard loader. Importing historical
two-output checkpoint files still uses `core.training.load_checkpoint`; the
single-logit legacy importer does not silently convert those files.

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
- `{lf,hf}_means.{png,pdf}` and `{lf,hf}_coverage.{png,pdf}`.
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

## Unequal event counts per voxel

Voxels may contain different numbers of events. Data preparation pads every voxel
of a split/fidelity group to the largest count and stores the true counts as
`event_counts` (`[B]`) in `StandardBatch.event_counts` and in
`metadata.json`. Padded positions have zero labels and features.
Groups with equal counts keep the previous rectangular format, so existing
prepared datasets and checkpoints are unaffected.

- **Training** weighs voxels equally: voxels are drawn uniformly and each episode
  samples context and targets only from that voxel's own events. `n_events` must
  fit the smallest training voxel. This holds for natural sampling, positive quota
  (pool prevalence uses the voxel's own events) and class-aware mixup. BDT fits all
  valid events and drops padding.
- **Context/target splits** (`split_context_target`, evaluation manifests) use a
  fixed context size; each voxel's target is its remaining events, so target
  counts differ per voxel and are carried as `event_counts` / `target_counts`.
- **Voxel means** (`EventPrediction.mean`, `StandardBatch.voxel_mean`, observed
  `m/N`) divide by each voxel's own count. Event-level metrics (log loss, Brier,
  AP) use valid events only.
- **MFGP**: `training_arrays.npz` stores `events_k` per voxel of level k, and log
  smoothing `(m + a)/(N + 2a)` uses each voxel's own N. The GP still has one
  constant noise per level, so voxels with smaller N are noisier than the model
  assumes. Observed-fraction projections
  need an explicit `projections.target_events` when validation counts differ.
- The legacy `core.training.train_cnp` loop, `cnp_trial_predictive`, `evaluate_bernoulli`, `score_bdt` and the legacy MFGP
  dataset helpers reject unequal counts with an error; use `core.surrogates`.

## Datasets and providers

Training, inference and the GP stage read data only through a dataset provider
(`core/surrogates/datasets.py`). A run config names it by import path; without one,
`NpzBatches` reads `batches/<partition>/<fidelity>.npz` (synthetic data, tests):

```yaml
data_directory: outputs/my_prepared_data
data_provider: my_package.provider:MyDataset    # constructed as MyDataset(data_directory)
data_config: my_data.yaml                        # optional: provider.prepare() rebuilds if outdated
partitions: {train: train, validation: validation, test: test}   # provider partition names
```

A provider implements `partitions()`, `load(partition, fidelity, labels=True)` returning
`(StandardBatch, provenance)` (with `labels=False` also for inputs without outcomes; the
labels are then a zero placeholder that is never scored), `theta(partition, fidelity)`
returning normalized theta and voxel ids, `coordinates(partition, fidelity)` and
`normalization()`. Optionally a classmethod `prepare(data_config, data_directory)` rebuilds
the prepared data and returns a record that the run stores in `experiment.json`. GP levels
and evaluation sets name provider partitions directly. Application packages ship their own
provider and configs.

Two settings make large validation sets cheap. `selection_max_voxels: N` selects
checkpoints during training on a seeded subset of at most N validation voxels; the full
validation and test sets are still evaluated once after training. `evaluation_threads: N`
runs evaluation passes with N torch CPU threads; results are identical.

## Context-free inference

```bash
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION --threads 8
```

writes `RUN/inference/best/<partition>_<fidelity>.{npz,json}`: per voxel the provider's voxel
id and coordinates, normalized theta, event count and `mean_probability` (mean predicted p
over the voxel's events), plus provenance. It uses the provider the checkpoint was trained
with (`--data-provider` overrides it). Only context-free models (mlp, transformer, bdt) are
accepted, and the provider normalization must equal the checkpoint's. Results are never
overwritten. Use voxels the model was not trained on, so no labels can be memorized.

### Log-space MFGP

`mfgp.output_transform: log` fits all three GP levels to natural logarithms;
`identity` remains the shared API default and reproduces linear-space behavior.
Observed-fraction levels use `(m+a)/(N+2a)` with `raw_pseudocount: 0.5` and each voxel's
event count. This Beta(1/2,1/2) posterior-mean
smoothing applies to all HF raw training fractions, including zeros. CNP means
are not counts: they only receive the numerical `cnp_log_floor: 1e-12`.

`training_arrays.npz` preserves original values; `fit_arrays.npz` records the
adjusted original-scale targets, `transformed_training_arrays.npz` records
exact logged targets, and `levels.json` records per model-mean level how many values hit
the floor (`log_floored`).
`MultiFidelityGP.predict` returns original-scale lognormal mean and variance;
`predict_transformed` returns Gaussian log-space moments. `predict_interval`
returns asymmetric equal-tail bounds. Save/load preserves these semantics;
older checkpoints default to identity mode.

Validation errors use unchanged raw fractions. Coverage plots use transformed
interval bounds, not mean ± original-scale standard deviation. Positive bounds
cannot cover a zero observed fraction: this diagnostic is not a binomial
posterior-predictive count interval. Lognormal predictions can exceed one.
Observation noise is learned in log space and is not CNP uncertainty.
Active-learning acquisition/refitting currently requires identity mode; the
offline training and prediction workflow supports both transforms.

### Spatial posterior marginalizations

`mfgp.projections.enabled: true` adds highest-fidelity spatial projections after
GP fitting; `ensure_selected_projections` also generates them from an existing saved
model, without retraining or loading raw data. The implementation follows
[RESOLVE's posterior-draw averaging recipe](https://github.com/annkasch/resolve/blob/4219c8c101d30acc38401465c5dc17ba88dcce1a/resolve/regression_models/bayes_pce_multi_fidelity_visualizer.py#L164),
not its single-prediction bin-percentile method, whose spread describes spatial
variation rather than uncertainty in a population average.

All three coordinates are queried jointly at the highest GP fidelity. Draws use
the full latent posterior covariance, excluding likelihood noise, conditional on
fitted hyperparameters. In log mode, each draw is exponentiated before averaging.
Uniform midpoint cells approximate physical-volume averages. Curves integrate
out two coordinates; plane maps integrate out one. Each remaining bin normalizes
by its valid cell count, so this is a conditional average, not an unnormalized
integral. Posterior quantiles of the averaged draws define 68.27%, 95.45%, and
99.73% pointwise equal-tail intervals (not simultaneous bands). The plotted mean uses analytic pointwise latent
moments averaged with the same weights. It is not the median. Observation-predictive
coverage plots remain separate and retain their previous noise convention.

Example projection settings (`mfgp.projections`):

```yaml
mfgp:
  projections:
    enabled: true
    domain: training_convex_hull
    grid_steps: 12
    n_draws: 8192
    seed: 42
    axis_labels: [x, y, z]
    units: m
```

The default domain is the convex hull of LF/HF **training** centers, mapped back
using the saved normalization. It is empirical support, not a detector geometry
model; it cannot represent concavities or holes. `domain: box` requires explicit
`bounds: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]` in physical units. A box must
lie within the intended volume. Optional bounds also restrict a hull domain.
No bounds are inferred from validation labels or validation positions. Grid
midpoints outside the domain are excluded; unsupported projected cells remain
blank, never filled with zero. Defaults are exploratory: check grid/draw
convergence before interpreting fine structure or extreme tails. Dense joint
covariance scales quadratically in valid grid points; grid_steps is capped at 16.

Artifacts under `RUN_DIR/mfgp/projections/`:

- `curves.png/pdf`: three coordinate curves with all three bands.
- `curves_observed.png/pdf`: the same curves with individual HF training and
  validation target fractions overlaid and identified separately.
- `planes.png/pdf`: xy/xz/yz means and three interval-width maps.
- `planes_observed.png/pdf`: mean maps with observed target fractions as colored
  markers, on the same color scale; training circles and validation diamonds.
- `projections.npz`: physical grids, domain mask, projected means/interval bounds,
  and unchanged raw target fractions/positions from the saved run.
- `metadata.json`: configuration, model/input hashes, numerical jitter, domain,
  weighting and uncertainty definitions. Matching artifacts are cached.

Overlaid observations are individual voxel measurements at different omitted
coordinates, not volume-marginalized measurements. Their scatter includes spatial
variation and counting noise. Their inclusion in the marginal bands is **not** a
coverage test. Three-sigma tail quantiles are Monte Carlo estimates, not exact.

### Predictive projections for observed voxel fractions (projection 3)

`mfgp.projections.quantity: observed_fraction` selects this mode; `latent_mean` remains available for uncertainty in spatial averages. Predictive
projections retain variation between locations: for each posterior draw and each
retained coordinate bin, uniformly select a valid grid location in the omitted
coordinates, then use a binomial count with its latent probability. The resulting
`m/N` samples determine discrete equal-tail 68.27/95.45/99.73% quantiles. No
spatial averaging occurs before these quantiles. The central curve remains the
population mean. Both 1D and plane projections support this mode.

`target_events: null` (default) recovers N from the saved evaluation set when all its
voxels have the same event count; otherwise set it explicitly. A different N describes a different simulation
budget and should not be interpreted as matched to the overlaid measurements.

Only latent GP uncertainty is sampled; fitted Gaussian observation noise is not
added alongside binomial noise. Treating the latent log-GP as a probability model
is approximate because fitting used smoothed rates. Any sampled probabilities
outside [0,1] cause an error rather than clipping. The log-GP does not enforce
probability support; substantial excursions need a bounded model.

These are **unconditional** predictions, allowing zero counts, over a uniform
spatial population in the configured domain. Current nonzero-selected HF files
and their spatial sampling do not represent that population exactly. Overlays
remain descriptive, not evidence of nominal predictive coverage. No rejection
sampling or conditioning on positive counts is silently applied.

Artifacts are written under `mfgp/projections_observed_fraction/`, keeping the
latent-mean artifacts separate. Grid/draw resolution and tail Monte Carlo error
remain relevant; discrete intervals may contain more than nominal probability.

Predictive observed-overlay figures include per-projection validation counters
for all three bands, alongside their nominal probabilities. Counts use interpolated displayed bounds for curves and the saved grid cell
for planes (last edge inclusive), with inclusive interval bounds. Points outside
projected edges, in undefined cells, or with nonfinite values are excluded and
reported; denominators count only evaluable validation points, never training
points. `coverage_counts.json` stores the counts for all six projections. These
are descriptive interval-inclusion rates, not proof of population calibration
for the spatially/nonzero-selected HF validation sample. No extrapolation is used.

The 1D predictive displays now connect grid-center estimates linearly; the first
and last values extend only to their own cell edges. Validation counters use
exactly those interpolated bounds, with no extrapolation beyond the projection
range. Plane counters retain grid-cell lookup. Observations use circle markers.
Points outside training-derived projection bounds remain visible but are excluded
from inclusion denominators. Rendering does not change the sampled predictions.
