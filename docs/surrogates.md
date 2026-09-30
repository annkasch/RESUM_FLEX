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
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m core.surrogates train config.optical.resum.yaml
```

Only `config.optical.resum.yaml` is retained as a run configuration. Change its
model and training settings using the API options below; BDT fits natural events. Change `selection` to `average_precision` to select
for event discrimination, or `bernoulli_log_loss` for event BCE. The default is
`voxel_rate_mae`. Change `output_directory` for every new run: existing nonempty
runs are never overwritten. Data/output paths resolve relative to the YAML file.

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

## MFGP boundary

```python
from core.surrogates import prepare_surrogate_datasets
from core import fit_mfgp_three_fidelity

arrays = prepare_surrogate_datasets(model, lf_batch, hf_batch, seed=0)
gp = fit_mfgp_three_fidelity(arrays)
```

The helper requires theta and averages predicted probabilities and HF labels over
the same target sets. It retains the historical `Y_lf_cnp` / `Y_hf_cnp` dictionary
keys for GP compatibility; these now hold scores from the selected surrogate.
The existing `prepare_mfgp_datasets_from_batches` also accepts an `EventSurrogate`.
Importing or using event models does not require GPy/Emukit. Actual GP fitting does.
Classifier uncertainty is not substituted for the GP posterior uncertainty.

## Existing trained models

```python
from core.surrogates import import_legacy_surrogate
model = import_legacy_surrogate("outputs/bce_5percent/seed0/stable_bce/best.ckpt")
model.save("outputs/imported_cnp")
```

Single-logit CNP/MLP/transformer `.ckpt` files and BDT `.joblib` files can be
adapted without retraining. Trees without saved feature metadata require explicit
`dim_theta` and `dim_phi`. Old two-output Gaussian CNP checkpoints keep their
existing `core.training.load_checkpoint` API; they are not silently converted.
All checkpoint loaders are for trusted local files.

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

Run `python -m core.surrogates train config.optical.resum.yaml`.
Change the model kind for MLP or transformer. Only CNP uses context observations;
other neural models use the same mixed targets. Context size is randomly chosen
within `n_context_min` / `n_context_max` each batch. Mixup defaults to real context.
Stable BCE (`focal_gamma: 0`) accepts soft target labels. BDT does not support
this sampler. Sampling correction and class weights are rejected with mixup;
no probability-bias correction is implied. Positive quotas cannot be combined
with it. Permutation mixup remains a standalone utility.

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

To try this objective, edit `training.objective` in `config.optical.resum.yaml`
using the settings above before running it.

## Legacy two-output CNP

`model.kind: legacy_cnp` wraps the existing `build_cnp` architecture without
changing its decoder or likelihood definitions. Set `training.loss` explicitly:

- `theory-truth`: Bernoulli NLL on the effective logit computed from both outputs.
- `practice-truth`: the original Gaussian NLL on the transformed mean and scale.

Single-logit models keep `training.loss: bernoulli` (the default). Mismatched
model/loss settings fail validation. Legacy losses currently require no extra
weights and `focal_gamma: 0`. Natural sampling, class-aware mixup alone, and
`objective.strategy: real_plus_mixup` are supported. For the combined objective,
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

The retained `config.optical.resum.yaml` uses the legacy model with standard
mixup. Select the combined objective explicitly using the settings above.

Run it with `python -m core.surrogates train config.optical.resum.yaml`. New checkpoints
roundtrip both output channels through the standard loader. Importing historical
two-output checkpoint files still uses `core.training.load_checkpoint`; the
single-logit legacy importer does not silently convert those files.

## Full optical RESuM pipeline

`config.optical.resum.yaml` trains the legacy two-output CNP on prepared LF
non-zero-hit voxels using standard class-aware mixup (`mix_context: true`),
`theory-truth` Bernoulli NLL and the single-branch objective. It then fits the
existing three-level MFGP on training data only: LF CNP means, HF CNP means,
and HF raw target fractions. Zero-hit LF and HF voxels are excluded. Of the
56 retained HF voxels, 10 are selected for training with seed 42 and 46 are
assigned to validation, including former HF test voxels. There is no separate
HF test set. The current all-LF test uses all 159 retained LF voxels for training
and skips LF validation/test. Validation represents the non-zero-hit
subset, not the full spatial population.

The prepared dataset is `outputs/optical_data_alllf_hf10_nonzero`. Its
`split_policy.json` records the selection and excluded files; `config.json`
and `splits/voxel_split.json` allow reconstruction via `prepare_optical_data`
with the saved assignments. Normalization is refitted using training files only.
The previous prepared dataset and model outputs are retained.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m core.surrogates train config.optical.resum.yaml
```

The optional `mfgp` run-config section enables this second stage. It requires the
GP dependencies. `kernel`, `n_restarts`, `seed`, and `n_context` are explicit.
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

## Data preparation and repository layout

The current workflow uses library components and the shared CLI. The temporary
BDT/MLP/transformer/mixup experiment scripts and notebooks have been removed.
Saved outputs remain available locally. Original synthetic examples under
`scripts/phase*` and the original repository notebooks remain unchanged.

Prepare optical data using the integrated library, after installing the
`optical-data` extra and setting the source directory in `config.optical.yaml`:

```python
from schemas.config import load_config
from data import prepare_optical_data

config = load_config("config.optical.yaml")
prepared = prepare_optical_data(config.data)
```

This reads original LH5 files and saves normalized batches and split metadata;
it does not run model training. To use an existing prepared dataset, point a
`config.surrogate.*.yaml` file or `config.optical.resum.yaml` at its directory.
Choose a new output directory for each training run.

## Full optical notebook

`notebooks/lar_optical_map.ipynb` is the single maintained optical workflow
notebook. It calls the shared `run_experiment` API using
`config.optical.resum.yaml`, without duplicating model or training code.

Use the repository Python environment as the kernel. Run All trains from scratch
by default, with a timestamped output directory. `RUN_TRAINING = False` loads
`EXISTING_RUN` instead and displays that run's saved configuration. The default
is a two-output CNP, theory-truth, standard class-aware mixup, mixed context,
and the three-fidelity MFGP. `LOSS_OVERRIDE = "practice-truth"` selects Gaussian
NLL for a new run. No real-plus-mixup auxiliary objective is enabled.

The notebook checks prepared non-zero-hit LF/HF batches, uses 10 HF modeling voxels,
shows CNP metrics/PR and MFGP coverage bands, displays the development-coordinate
map, and leaves all checkpoints and prediction arrays in the run directory. It does
not read test files, regenerate simulation data, or submit a Slurm job.
