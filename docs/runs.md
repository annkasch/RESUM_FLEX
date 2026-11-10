# Runs and outputs

CLI stages, what a run writes, compute accounting and context-free inference.

## One experiment entry point

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 1. Train an event model (data preparation runs automatically from data_config)
.venv/bin/python -m core.surrogates train TRAINING.yaml
# 2. Mean predicted p per design point of a partition (context-free models; labels not needed)
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION
# 3. Fit the multi-fidelity GP on its configured levels
.venv/bin/python -m core.surrogates mfgp GP.yaml
```

Data come from a dataset provider ([data.md](data.md#datasets-and-providers)).
Change model and training settings using the API
options in [models.md](models.md) and [training.md](training.md); BDT fits natural events. `selection` picks the checkpoint metric
(`bernoulli_log_loss` for natural-event runs, `average_precision` for mixup ablations).
Change `output_directory` for every new run: existing nonempty runs are never
overwritten. Data/output paths resolve relative to the YAML file.

The runner loads prepared LF training and LF/HF validation batches. Inputs must
already have the training-only normalization applied. It never reads test data.
`train_sets` replaces the LF training partition by several sets, e.g. LF2 + HF train
for a model that is benchmarked against the GPs on their own training data; their design points
form one zero-padded batch, and the training split is labelled by their common fidelity
(`mixed` otherwise). LF validation selects the model by default; HF is then scored after
selection (`selection_fidelity` changes this). Training PR is in-sample. Files under the output directory are uniform across model families:

- `experiment.json`: resolved configuration, input-file hashes, normalization,
  feature metadata, split manifest when available and `evaluation_targets` (`all_events`
  for context-free models, `after_context_pool` for context models; `docs/evaluation.md`).
- `checkpoints/{best,final}/`: versioned model manifest and backend weights.
- `history.json`, `sampling_audit.json`, `metrics.json`.
- `{lf,hf}_{train,validation}_{best,final}.npz` where evaluated: logits, labels,
  design point means, residuals, PR arrays and the per-design-point within-point gain terms;
  held-out splits add the gain null bands, the feature-calibration bins and, with
  `residual_test:`, the residual-test arrays (`docs/evaluation.md`). Only LF training is evaluated.
- Added later by `python -m core.surrogates within_point` (runs from before the within-point
  metrics): `within_point.json`, `{fidelity}_{split}_{checkpoint}_within_point.npz` and
  `_within_point_gain.png`; the run's own files stay unchanged.
- Shared PNG/PDF validation-history, mean/residual and PR plots; per held-out split the
  within-design-point gain and the feature-calibration plots.

The Python entry point is `core.surrogates.experiment.run_experiment(config)`.
Notebooks can call it and display artifacts, without duplicating a training loop.
The CLI limits Torch threads to one; BDT fitting limits its thread pools to one.
CUDA is opt-in through `training.device`; default examples run on CPU.

## Compute accounting

Every run records what it cost (`core.compute`): a training run, the binomial GP and the
MFGP write `compute.json` into their output directory, and context-free inference adds a
`compute` entry to its `.json` provenance. A combined run's `mfgp/` folder has its own file;
the parent also lists it as stage `mfgp`.

- `stages`: one record per stage. Training runs: `data_preparation`, `data_loading`,
  `training`, `last_layer_fit` (if configured), `evaluation`, `mfgp` (combined runs). GP
  runs: `data_preparation` (standalone), `data_loading`, `fit`, `evaluation`. Inference:
  `data_loading`, `prediction`.
- `total`: the whole run, including anything between stages.
- `environment`: host, Slurm job id, allocated logical CPUs and physical cores, GPUs and the
  thread variables.

Each record has `wall_seconds`, `cpu_seconds` (CPU time used, summed over threads),
`allocated_cores` (physical cores of the process's CPU affinity, i.e. the Slurm allocation),
`core_hours = wall_seconds × allocated_cores / 3600` (what the allocation costs, used or
not), `cpu_utilization = cpu_seconds / (wall_seconds × allocated_cores)` and the Torch
threads. The cost of a pipeline is the sum over its runs, e.g. event-model training +
inference + GP fit for NN + GP.

## Context-free inference

```bash
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION --threads 8
```

writes `RUN/inference/best/<partition>_<fidelity>.{npz,json}`: per design point the provider's design point
id and coordinates, normalized theta, event count and `mean_probability` (mean predicted p
over the design point's events), plus provenance. It uses the provider the checkpoint was trained
with (`--data-provider` overrides it). Only context-free models (mlp, field, transformer, bdt)
are accepted, and the provider normalization must equal the checkpoint's. Results are never
overwritten. Use design points the model was not trained on, so no labels can be memorized.
With a Bayesian last layer the file also holds `probability_draws` [S, B] and their
`probability_mean` / `probability_sd` per design point.

`--event-logits` (off by default) also writes every event's logit to
`RUN/inference/best/<partition>_<fidelity>_events/`: float32, padding dropped, in the
provider's event order (design point by design point, each design point's events in load order). Shards
`shard_NNNNN.npz` (`logits` [E], `event_counts` [b], `design_point_ids` [b]) hold whole design points, up to
2²² events each unless one design point alone has more; `index.json` lists each shard's design point range
and event count. `core.surrogates.inference.read_event_logits(directory, design_points=(start, stop))`
reads them back, opening only the shards it needs. With a Bayesian last layer these are the
MAP logits. Not combinable with `--events-from`.
