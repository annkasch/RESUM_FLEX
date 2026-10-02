# Controlled experiments

Use YAML to choose a dataset, model, training stages and evaluation protocol.
The same runner manages event-model pipelines and direct count-GPs. Existing
surrogate and count-GP YAML files are accepted through adapters; existing notebook
entrypoints remain supported. Numerical losses and likelihoods are unchanged.

## Start here

Open `notebooks/experiments.ipynb`. Select a configuration, inspect its resolved
settings, and set `RUN_EXPERIMENT = True` to launch. The default is inspection only.
The notebook browses the local catalog and displays saved reports. It contains no
training, sampling, metric or plotting implementation.

From the repository root, using its Python environment:

```bash
python -m core.experiments validate experiments/optical_fine_tuning.yaml
python -m core.experiments run experiments/optical_fine_tuning.yaml
python -m core.experiments list --store outputs/experiments
python -m core.experiments inspect RUN_ID --store outputs/experiments
python -m core.experiments report RUN_ID --store outputs/experiments
python -m core.experiments compare experiments/optical_mfgp_comparison.yaml
python -m core.experiments compare-saved RUN_A RUN_B --store outputs/experiments
python -m core.experiments rebuild --store outputs/experiments
python -m core.experiments import PATH_TO_OLD_RUN --store outputs/experiments
```

`validate` checks schemas, sources and checkpoint compatibility without training.
Dataset-dependent evaluation checks also run before model fitting. A runtime
failure leaves a failed run, traceback and any completed artifacts in the catalog.

## Configuration and extension points

- `dataset.prepared` selects a prepared-data directory. Alternatively,
  `dataset.preparation` selects an optical preparation YAML; `format: events`
  supports ordinary optical and cross-budget transfer data, while `format: counts`
  supports optical count preparation. Preparation is isolated and preserves
  existing split assignments without modifying their original manifest.
- `pipeline.kind: event` uses the existing `model` and `training` specifications.
  Alternatively provide `stages`, each with `name`, `start_from`, `trainable`,
  `training` and `selection`. `start_from` is `initial` or an earlier `name/best`
  or `name/final`. Each stage creates a fresh optimizer. Frozen output-layer
  stages disable dropout in frozen features. Sampling, weights and losses are
  resolved and saved separately for every stage.
- `initial_checkpoint` initializes fitting from an existing model. With
  `fit: false`, it evaluates that checkpoint and optionally fits the spatial GP,
  without retraining the network or copying it under another checkpoint name.
- `spatial_regression` uses the existing MFGP settings. `spatial_checkpoint`
  explicitly selects `best`, `final`, `pretraining` or `stages/NAME/best|final`.
- `pipeline.kind: count_gp` uses `backend`, prediction draw settings and
  `projections`. Its training data are full counts from the training partitions;
  LF/HF label the reporting groups, not different physical latent functions.
- `evaluation` fixes partitions, context size, evaluation seed and plot selection.
  Weights are currently equal per voxel. Event metrics weight real events equally.
  Selection uses LF validation only, never a test partition.

Existing BDT training remains all-natural-event fitting with its supported weights;
unsupported neural freezing or continuation requests fail validation. Supporting a
new algorithm requires its model/backend adapter, configuration and capability
checks. Ordinary changes to existing models or schedules require only YAML.

## Controlled comparisons

A comparison references one baseline and named nested configuration overrides.
Lists replace lists; changing a discriminated `kind` replaces that component.
Paths in overrides are relative to the comparison YAML. Dataset, evaluation and
store cannot change within a controlled comparison. The first variant is the
paired baseline. Explicit `seeds` control training and GP fitting; evaluation seeds
stay fixed. Stage-seed offsets are preserved. Downstream-only experiments vary GP
seeds without retraining the saved CNP.

`optical_mfgp_comparison.yaml` repeats the before/after experiment using the actual
saved checkpoints from the October run. Change those paths to compare another run.
Both arms share snapshots and evaluated targets. The output includes resolved
child configurations, declared differences, paired metric deltas and per-seed
mean/standard deviation. A single seed has no estimated between-seed standard
deviation. Matching seeds do not guarantee bitwise identical numerical optimization.

Saved-run comparisons separate different target-manifest groups rather than rank
incompatible runs. Population metrics can compare event score averages, MFGP
predictions and count-GP predictions on matching targets. Event ranking exists only
for event models. Distribution scores retain their predictive interpretation;
continuous-density NLL and count-mass NLL are never pooled. No automatic winner
selection uses test results.

## Common targets and statistical interpretation

Event datasets save each evaluation voxel's context/target event IDs, row indices,
hit counts, denominators, physical coordinates and weights. Counts derived from the
same event snapshot use the **same held-out target events** for evaluation. The count
GP does not condition on context labels; the CNP's access to context is explicit in
its prediction record. Their training inputs also differ by design, and should not
be described as identical learning problems.

Aggregate-only count files cannot reconstruct event IDs. They require
`context_events: 0` and are marked `aggregate_only`; they are not silently equated
with event-based evaluations. Missing event IDs in synthetic/prepared datasets are
identified as prepared-row indices rather than claimed simulator identities.

Exact-coordinate distribution coverage, projected inclusion, and uncertainty on a
marginalized mean remain distinct. Projected axes/planes are the default projection
choices; marginalizations are opt-in. Invalid probability-range predictions are
counted, not clipped or hidden. This framework does not repair the existing
lognormal MFGP behavior or establish calibration.

## History, immutability and reproducibility

`outputs/experiments/datasets/HASH` contains deduplicated copies of prepared arrays
and metadata. Raw simulations are referenced by their existing preparation
provenance; they are not duplicated. Source changes produce another snapshot.
Snapshots and completed-run artifacts are hash-verified by the framework.

Each `runs/RUN_ID` contains `run.json`, resolved configuration, code/environment
provenance, dataset reference, evaluation manifest, checkpoint dependencies,
backend artifacts, shared prediction records, metrics, report and captured warnings.
Provenance records Git commit, tracked implementation patch, untracked Python
implementation files, package/device/thread information and seeds. It does not
capture arbitrary environment variables or duplicate entire working directories.

Completed manifests are terminal. Regeneration writes a new report under
`reports/RUN_ID/REPORT_ID`, never into the completed run. The SQLite catalog is only
an index; `rebuild` reconstructs it from manifests. A crashed process can leave a
`running` record; inspect its artifacts rather than treating it as completed.

Imported historical runs are copied and labeled with unknown missing provenance.
They appear in history but are excluded from automatic native comparisons until
an explicit evaluation manifest and prediction records exist.
