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
.venv/bin/python -m core.surrogates train config.optical.centered.mlp.yaml
# 2. Mean predicted p per unlabeled voxel (context-free models)
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR gp_unlabeled
# 3. Fit the multi-fidelity GP on its configured levels
.venv/bin/python -m core.surrogates mfgp config.optical.centered.mfgp.yaml
```

On Perlmutter submit each step with
`sbatch scripts/slurm/train_surrogate.sbatch CONFIG.yaml [train|mfgp]`.

The configs for the centered dataset (`config.optical.centered.*.yaml`) are the
maintained examples: `cnp.data` prepares the data, `cnp`/`mlp`/`cnp.mixup` train event
models, `mfgp`/`mfgp.log` fit the GP. Change model and training settings using the API
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

The optical configs use the single-logit `cnp` since 2026-10-02. Under
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
validation MAE. The current optical test sets `lf_validation: false`: it uses
the final CNP checkpoint after all configured steps, without selecting on HF.
LF validation files and diagnostics are skipped; LF training PR and training
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
  assumes (see `docs/stats_review.md`, finding 3). Observed-fraction projections
  need an explicit `projections.target_events` when validation counts differ.
- The legacy `core.training.train_cnp` loop, `cnp_trial_predictive`, `evaluate_bernoulli`, `score_bdt` and the legacy MFGP
  dataset helpers reject unequal counts with an error; use `core.surrogates`.

## Data preparation and repository layout

The current workflow uses library components and the shared CLI. The temporary
BDT/MLP/transformer/mixup experiment scripts and notebooks have been removed.
Original synthetic examples under `scripts/phase*` and the original repository
notebooks remain unchanged.

Prepare optical data using the integrated library, after installing the
`optical-data` extra and setting the input folders in a data config:

```python
from schemas.optical import load_optical_config
from data import prepare_optical_data

config = load_optical_config("config.optical.centered.cnp.data.yaml")
prepared = prepare_optical_data(config)
```

This caches new or changed LH5 files (see "Shared event cache"), assigns the split, fits
normalization and saves a small view plus split metadata; it does not run model training.
Training and GP configs point at the prepared directory (`data_directory`) and, with
`data_config`, regenerate it when the data config or raw files change.
Choose a new output directory for each training run.

## Role sources (event models and GP)

`source.sources` lists folders (`fidelity: lf` by default, or `hf`) and how each folder's
voxels are divided among the roles `cnp_train`, `cnp_validation`, `cnp_test` (event models),
`gp_train`, `gp_unlabeled` (inputs only), `gp_validation` and `gp_test` (GP). Each folder
gives either `fractions` (sum to 1, rounded per folder) or `counts` (file counts; one role may
take `rest`). Roles left out get no files, so an already split folder is a single role at 1.0:

```yaml
source:
  sources:
  - path: /data/lower_cyl/lf
    fractions: {cnp_train: 0.6, cnp_validation: 0.1, cnp_test: 0.0, gp_train: 0.3}
  - path: /data/lf_batch2
    counts: {cnp_train: 5000, cnp_validation: 1000, gp_train: rest}
  - path: /data/lf_val_only
    fractions: {cnp_validation: 1.0}
  - path: /data/hf
    fidelity: hf
    fractions: {gp_train: 1.0}
split:
  seed: 42
  manifest: outputs/my_data/splits/voxel_split.json
```

Splitting keeps whole voxels together and is seeded per folder; the global split fractions
are rejected in this layout. A voxel position shared by different roles is an error, except
between `gp_train` and `gp_unlabeled` (nested multi-fidelity design: HF voxels at positions of
LF inputs). Normalization is fitted on `cnp_train` and applied to every role and fidelity.
`python -m core.surrogates train` reads `cnp_train`, `cnp_validation` (with
`lf_validation: true`) and `cnp_test` (with `test_fidelities: [lf]` and `unblind_test`). Set
`hf_validation: false`; the HF roles are GP levels and evaluation sets (`mfgp` levels name the
role partitions directly, e.g. `gp_train`/`hf`).

Two settings in the training config make large validation sets cheap.
`selection_max_voxels: N` selects checkpoints during training on a seeded subset of at most N
LF validation voxels (seed `validation_seed`); the full validation and test sets are still
evaluated once after training. `evaluation_threads: N` runs evaluation passes with N torch CPU
threads and restores the training thread count afterwards; results are identical. Measured:
2.65 ms per 1000-event voxel on 1 thread, 0.9 ms on 16.

Set `data_config: <data YAML>` in the training config to keep the prepared data in sync.
Each `core.surrogates train` run then compares the data YAML with `config.json`, which only
preparation writes, and the raw files with `source_stamp.json` (path, size and modification
time of every file). If both match, the raw files are not read. Otherwise the run checks file
contents and regenerates the prepared data if anything differs, before training starts.
The data YAML's `output_directory` must be the training `data_directory`. Whether data was
regenerated is recorded in the run's `experiment.json`.

## Unlabeled inputs and context-free inference

Simulations without optical output (vertex and momentum only, e.g. `LF_primaries`) go into
the role `gp_unlabeled`. The event cache stores them with `labeled = False`; reading their
labels or hits raises, and preparation fails if such a file lands in a labeled role. Their
partition loads only through `OpticalView(dir).load_inputs(partition, fidelity)`, whose
labels are a zero placeholder for the `StandardBatch` contract and must never be scored.

```bash
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR gp_unlabeled --threads 8
```

writes `RUN/inference/best/gp_unlabeled_lf.{npz,json}`: per voxel the file, nominal
center, normalized theta, event count and `mean_probability` (mean predicted p over the
voxel's events), plus provenance. Only context-free models (mlp, transformer, bdt) are
accepted, and the view normalization must equal the checkpoint's. Results are never
overwritten. Use voxels the model was not trained on, so no labels can be memorized.

## Shared event cache

Raw LH5 files are read once into a cache shared by all data configs: by default
`<output_directory parent>/event_cache`, or `cache_directory` in the data YAML. Per file it
keeps a float32 copy of each event's vertex and momentum (`[n, 6]`), the any-channel label,
and one hit-list row per (event, detector) with the number of hits. Coordinates and other
features are computed when a partition is loaded (about 0.2 µs per event), so changing feature
settings never rewrites the cache.

- `index.npy` has one row per file: folder, name, size, mtime, sha256, voxel centers, event
  and hit counts, and the largest `n_part`. `cache.json` lists the folders and shards.
- Each update reads only new or changed files (by size and mtime). Worker processes, one per
  available CPU, each read a share of the files and write one shard. A changed file goes into
  a new shard and its old row is marked invalid.
- `python -m data.event_cache update DATA.yaml [--workers N]` fills the cache without preparing
  (run it on a compute node for large inputs). `python -m data.event_cache compact CACHE_DIR`
  merges the shards of many small updates and drops invalid rows.
- One writer at a time: updates take a `.lock` directory. If a writer crashed, remove the lock
  by hand.

A prepared dataset is a view on the cache: `view.npy` lists each file's partition, fidelity,
path and sha256, and `view.json` stores the cache directory, data config and normalization.
`load_partition(directory, partition, fidelity)` (in `data.optical_pipeline`) builds the
normalized `StandardBatch` (phi in float32) and returns its provenance: a sha256 over the
selected files' contents and `view.json`. Loading checks that the cache still holds the same
file contents. Normalization is fitted with streaming moments over the training files, so
the training events never need to be in memory at once.

## Multiple raw input folders

A data config can list several folders per
fidelity under `source.directories.lf` and `source.directories.hf`. Each folder
must directly contain `.stp.lh5` files; fidelity comes from the configuration,
so folder names need not be `lf` or `hf`. For example:

```yaml
source:
  directories:
    lf: [/path/to/lf1500_part1, /path/to/lf1500_part2]
    hf: [/path/to/hf_part1, /path/to/hf_part2]
```

The previous `source.directory` form (one root containing `lf/` and `hf/`)
remains supported. Specify one form only. Relative paths are resolved against
the preparation YAML. Repeated references/symlinks to the same physical file
are loaded once. Different files with the same fidelity and basename, and the
same physical file assigned to both fidelities, are rejected as ambiguous.
Splitting and training-only normalization happen after collecting all folders.
Event counts may differ between voxels (see "Unequal event counts per voxel");
mixing primary budgets within LF remains a separate, retired modeling choice.

## Automatic preparation

A training or GP config with `data_config` calls `ensure_prepared_optical_data` before
loading data. The data config's output directory must equal `data_directory`. The helper checks raw-file fingerprints,
configuration, manifest, and the view; a missing or outdated view and normalization
are rebuilt, reading only files missing from the event cache. Unchanged valid inputs are reused. Changes
in source membership, file contents, or configuration automatically update the
manifest. Compatible existing assignments are retained; explicit split policies
take precedence. Invalid inputs and conflicting voxel groups still raise errors.

Preparation matches `stp/optical/evtid` to `vtx/evtid`, making one binary target
per simulated event: one if any configured detector recorded a hit, zero otherwise.
Multiple recorded hits for an event count once. This is an observed simulation
outcome, not a model-assigned label. Current selected inputs all have one primary
per event (`vtx/n_part == 1`). Theta is the filename center; phi contains vertex
offsets from that center and momentum components. Standardization is fitted on
training files only, then applied to both training and validation. Prepared
arrays, normalization, source fingerprints, split metadata and reports are saved.


## Folder names describe simulation statistics

The canonical input layout under
`/global/cfs/cdirs/m2676/users/aschuetz/data/lar_optical_map/by_primaries/` is:

```text
N500/    nonzero/    zero/
N750/    nonzero/    zero/
N1000/   nonzero/    zero/
N1500/   nonzero/    zero/
N5000/   nonzero/    zero/
```

`N` denotes simulated primary count. Each category contains symlinks to the
original `.stp.lh5` files. Counts (nonzero / zero) are 38/47, 157/70, 146/78,
159/65 and 56/9 respectively. The root manifest records every source and excludes
eight failed one-primary simulations. Original source folders are preserved.

Folders have no inherent LF/HF role: `source.directories` assigns that role when
preparing a dataset. To include zero-hit voxels, list the matching `zero` folder too. Preparation accepts unequal
event counts per voxel; this layout does not by itself reintroduce training pooled
across different LF primary counts.

### Log-space MFGP

`mfgp.output_transform: log` fits all three GP levels to natural logarithms;
`identity` remains the shared API default and reproduces linear-space behavior.
The optical configuration enables `log`. No CNP retraining behavior changes.
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
offline optical-map training and prediction workflow supports both transforms.

### Spatial posterior marginalizations

`mfgp.projections.enabled: true` adds highest-fidelity spatial projections after
GP fitting. The optical notebook also generates them from an existing saved
model, without retraining or loading current raw data. The implementation follows
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

The optical config now selects `mfgp.projections.quantity: observed_fraction`.
`latent_mean` remains available for uncertainty in spatial averages. Predictive
projections retain variation between locations: for each posterior draw and each
retained coordinate bin, uniformly select a valid grid location in the omitted
coordinates, then use a binomial count with its latent probability. The resulting
`m/N` samples determine discrete equal-tail 68.27/95.45/99.73% quantiles. No
spatial averaging occurs before these quantiles. The central curve remains the
population mean. Both 1D and plane projections support this mode.

`target_events: null` (default) recovers N from the saved HF validation artifact.
New runs store it explicitly. For older runs, recovery uses the fingerprint-checked
prepared validation batch and saved context count, or requires an explicit
`target_events` override if those inputs are unavailable. Current optical settings
use 5000 - 64 = 4936, not 5000. A different N describes a different simulation
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
