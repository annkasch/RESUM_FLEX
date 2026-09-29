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
  .venv/bin/python -m core.surrogates train config.surrogate.cnp.yaml
```

Four example configs are provided: `config.surrogate.{cnp,mlp,transformer,bdt}.yaml`.
The neural examples retain stable BCE and corrected 5% positive sampling; the
BDT fits all natural events. Change `selection` to `average_precision` to select
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

Run `python -m core.surrogates train config.surrogate.cnp.mixup.yaml`.
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
