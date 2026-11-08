# Config settings

Every YAML file is validated by a pydantic model in `schemas/surrogates.py`; unknown keys
are rejected. Relative paths resolve against the YAML file. Defaults are given in
parentheses.

- [Training config](#training-config) (`python -m core.surrogates train`)
- [GP config](#gp-config) (`python -m core.surrogates mfgp`, or `mfgp:` in a training config)
- [Binomial GP config](#binomial-gp-config) (`python -m core.surrogates binomial`)
- [Residual test config](#residual-test-config) (`python -m core.surrogates residual`)
- [Within-design-point config](#within-design-point-config) (`python -m core.surrogates within_point`)

## Training config

`SurrogateRunConfig`.

### Data and output

| Key | Meaning |
|---|---|
| `data_directory` | Prepared data, read by the dataset provider. Required. |
| `data_provider` | Provider as `module:Class` (none: `.npz` batches, `batches/<partition>/<fidelity>.npz`). |
| `data_config` | Data-preparation YAML for the provider; the data are rebuilt when it or the raw files change (none). |
| `partitions` | Provider partitions for `train`, `validation`, `test` (each named like the key). |
| `train_sets` | Train on these sets instead of `partitions.train` at LF: a list of `{partition, fidelity}`, e.g. the GP's training data `[{partition: gp_unlabeled, fidelity: lf}, {partition: gp_train, fidelity: hf}]` for a model that predicts voxel rates itself. Their voxels form one zero-padded batch; the validation and test partitions are not allowed (none). |
| `output_directory` | Run directory. Required; must be new or empty. |

### Evaluation

| Key | Meaning |
|---|---|
| `selection` | Checkpoint metric on the selection set: `bernoulli_log_loss`, `average_precision`, `voxel_rate_mae`, `voxel_count_nll` (`voxel_rate_mae`). Use log loss or `voxel_count_nll` (plug-in binomial count NLL per voxel, much more sensitive with few hits) for natural-event runs, AP for mixup runs. |
| `lf_validation` | Load and report LF validation; without a loaded selection set the final checkpoint is kept (`true`). |
| `hf_validation` | Load and report HF validation (`true`). |
| `selection_fidelity` | Validation fidelity that selects the checkpoint: `lf` or `hf` (`lf`). With `hf` that set's scores are selection-biased; keep a blinded test set for the final comparison. |
| `test_fidelities` | Test sets that exist, e.g. `[lf]` (`[]`). |
| `unblind_test` | Evaluate the test sets. Leave `false` until all choices are final (`false`). |
| `validation_context_events` | Context events per voxel for context models during evaluation (64). |
| `validation_context_sweep` | Extra context sizes evaluated after training, on the same targets (`[]`). |
| `validation_seed` | Seed of the evaluation context/target split (12345). |
| `selection_max_voxels` | Select checkpoints on a seeded subset of this many validation voxels (all). |
| `evaluation_threads` | CPU threads for evaluation passes (1). |
| `within_point_null_draws` | Label draws per design point for the within-design-point null (1000, ≥ 100). The ±3σ coverage is rough at 1000; 10 000 costs ~10×. Also in stream configs. |
| `feature_calibration` | Calibration per input column on every held-out split: `{bins: 10}`, equal-count bins per column (≥ 2). Also in stream configs (validation events and evaluation sets). |
| `residual_test` | Residual learnability test on every held-out split, settings below (none: off). Needs the `lgbm` extra; the run fails at start without it. |

### Residual learnability test (`residual_test`)

`ResidualTestConfig`, in training and stream configs (`docs/evaluation.md`).

| Key | Meaning |
|---|---|
| `folds` | Cross-fitting folds, split by design point (stream records: by event) (5). |
| `min_positives` | Fewer positives: not run, metrics `None` (50). |
| `num_leaves` | Leaves per tree (7). |
| `learning_rate` | LightGBM learning rate (0.05). |
| `min_data_in_leaf` | Minimum events per leaf (200). |
| `lambda_l2` | L2 penalty on leaf values (1.0). |
| `max_trees` | Upper bound on trees; the count is set by inner early stopping (500). |
| `patience` | Early-stopping rounds without improvement (20). |
| `threads` | LightGBM threads (1). |
| `max_positives` | Stream validation events only: keep a seeded uniform fraction `min(1, max_positives / positives)` of all events, so about this many positives remain; same fraction for hits and non-hits, so no weights; `null` = all (20000). A changed value is rejected in training configs. |

### Model (`model`)

`kind` selects the model; the remaining keys depend on it.

| `kind` | Keys |
|---|---|
| `cnp` | `encoder`: `latent_dim`, `hidden_dims`, `dropout` (64, `[128, 128]`, 0). Uses context events. |
| `mlp` | `architecture`: `hidden_dims`, `dropout` (`[128, 128]`, 0). |
| `field` | Neural field for point-cloud data (3-D direction). `architecture` as mlp; `position.phi_columns` (added to theta in physical units), `direction.phi_columns` + `degree` (4), `encoding`: `n_frequencies`, `max_frequency` (8, 8.0). See [models.md](models.md#neural-field-kind-field). |

`mlp`, `cnp` and `field` take an optional `last_layer`: `prior_precision` (`evidence` or a number), `draws` (1000), `seed` (0): a Laplace posterior over the final layer, fitted after training ([last-layer.md](last-layer.md)).
| `transformer` | `architecture`: `architecture` (`legacy` or `ft_transformer`), `token_dim`, `n_heads`, `n_layers`, `feedforward_dim`, `dropout`, `inference_chunk_size`. |
| `bdt` | `architecture`: `learning_rate`, `max_iter`, `eval_every`, `max_leaf_nodes`, `max_depth`, `min_samples_leaf`, `l2_regularization`, `max_bins`, `seed`. Needs `training.backend: bdt`. |
| `legacy_cnp` | Original two-output CNP; needs `training.loss: theory-truth` or `practice-truth`. |

### Training (`training`, neural models)

| Key | Meaning |
|---|---|
| `backend` | `neural` (or `bdt`, see below). |
| `n_steps` / `n_passes` | Run length: steps, or passes over the training events. Set one (10000 steps). |
| `eval_every` | Steps between validation passes (250). |
| `learning_rate` | Adam learning rate (0.001). |
| `batch_size` | Voxels per step (16). |
| `n_events` | Events drawn per voxel per step (128). |
| `seed` | Training seed (0). |
| `device` | `cpu` or `cuda` (`cpu`). |
| `grad_clip` | Gradient-norm clip, `null` to disable (1.0). |
| `loss` | `bernoulli`; `theory-truth`/`practice-truth` only for `legacy_cnp` (`bernoulli`). |
| `focal_gamma` | Focal-loss exponent; 0 is plain cross-entropy (0). |
| `targets` | Context models: `disjoint` scores only non-context events, `joint` also the context (`disjoint`). |
| `context` | Context size per step. `size: random`: drawn uniformly from `n_min` to `max_fraction` × `n_events` (16, 0.5), or up to an absolute `n_max` instead. `size: fixed`: always `n`. |
| `sampling` | `strategy`: `natural`, `positive_quota` (with `positive_fraction`) or `class_aware_mixup` (with `mixup`: `alpha`, `mix_context`) (`natural`). Only `natural` keeps the outputs calibrated probabilities. |
| `weighting` | `strategy`: `none`, `sampling_correction` (required with `positive_quota`) or `class_weights` (with `positive`, `negative`) (`none`). |
| `objective` | `strategy: single`, or `strategy: real_plus_mixup` with `mixup_loss_weight` and `real_target_ratio` (`single`). |
| `fine_tuning` | Second stage on the output layer: `n_steps`, `learning_rate`, `eval_every`, `n_events` (none). |

With `backend: bdt` only `weighting` applies (`none` or `class_weights`).

### Stages (`stages`)

Optional list of training stages run in order, instead of `training`'s single run. Each has
`name`, `start_from` (`initial` or `<earlier stage>/best|final`), `trainable` (`all` or
`output_layer`), `training` and `selection`. `initial_checkpoint` starts the first stage
from a saved model.

### GP after training (`mfgp`)

A GP block (see [GP config](#gp-config)) fitted right after training with the checkpoint
`mfgp_checkpoint` (`best`); none by default.

## GP config

`MFGPRunConfig` (standalone) or the `mfgp:` block of a training config (`MFGPStageConfig`;
there `mfgp_checkpoint` picks the event model, `best` by default).

### Levels (`levels`)

Ordered from the lowest to the highest fidelity; the last level is the target. At least two.

| Key | Meaning |
|---|---|
| `source` | `model_mean`: mean predicted p per voxel of an event model. `observed_fraction`: simulated hit fraction m/N. |
| `partition` | Provider partition of the level's voxels. |
| `fidelity` | `lf` or `hf` (`lf` for `model_mean`, `hf` for `observed_fraction`). |
| `inference` | `model_mean` only: file from `core.surrogates predict`; without it the event model predicts at fit time. |
| `max_voxels` | Fit on a seeded subset of this many voxels (all). |

### Evaluation and fit

| Key | Meaning |
|---|---|
| `evaluation` | List of `{partition, fidelity, level}` scored after fitting; `level` defaults to the target. |
| `output_transform` | `identity` (rate) or `log` (`identity`). |
| `raw_pseudocount` | Log mode: observed fractions become (m + a)/(N + 2a) (0.5). |
| `coverage_ylim` | Fixed y range `[low, high]` of the coverage plots (observed fraction); unset = from the data. |
| `cnp_log_floor` | Log mode: floor for model means (1e-12). |
| `kernel` | `rbf` or `matern52` (`rbf`). |
| `n_restarts` | Hyperparameter optimizer restarts (5). |
| `seed` | Seed of fits, voxel subsets and context splits (0). |
| `n_context` | Context events when a context model computes a level (64). |
| `unblind_test` | Allow test partitions in `evaluation` (`false`). |
| `projections` | Spatial projections of the target level: `enabled`, `plots`, `quantity`, `domain`, `bounds`, `grid_steps`, `n_draws`, … (disabled). |

### Standalone only

| Key | Meaning |
|---|---|
| `data_directory`, `data_provider`, `data_config` | As in the training config. |
| `checkpoint` | Event model for `model_mean` levels without `inference`. |
| `output_directory` | GP run directory; must be new. Required. |

## Binomial GP config

`BinomialGPRunConfig`: one latent hit probability fitted directly to per-voxel hit counts,
m ~ Binomial(N, sigmoid(f(θ))) with a Laplace posterior. No event model is involved.

| Key | Meaning |
|---|---|
| `observations` | List of count sets, each `partition`, `fidelity` and `max_voxels` (all). All sets share one latent probability; N enters the likelihood exactly. |
| `evaluation` | List of `{partition, fidelity}` scored with the observation-predictive distribution (`fidelity` defaults to `hf`). |
| `kernel` | `matern52` or `rbf`, plus a constant kernel (`matern52`). |
| `n_restarts` | Hyperparameter optimizer restarts (2). |
| `max_iters` | Optimizer iterations per restart (300). |
| `seed` | Seed of fits, voxel subsets and predictive draws (42). |
| `prediction_draws` | Monte Carlo draws of the predictive counts per evaluation set (32768). |
| `coverage_ylim` | Fixed y range `[low, high]` of the coverage plots (observed fraction); unset = from the data. |
| `covariate` | Optional `{inference: {"partition/fidelity": file}}` of `python -m core.surrogates predict`, one per observation and evaluation set: logit p = b logit(p_model) + c + delta(theta), b and c marginalized as linear/bias kernel terms. The event model must not be trained or selected on these sets. |
| `unblind_test` | Allow test partitions in `evaluation` (`false`). |
| `data_directory`, `data_provider`, `data_config` | As in the training config. |
| `output_directory` | Run directory; must be new. Required. |

## Residual test config

`ResidualRunConfig`: the residual learnability test on a finished training or stream run,
from the logits and labels it saved (`docs/evaluation.md`). The inputs are reloaded with
the run's resolved config; a set whose provenance (e.g. file hash) or labels differ from
the run's record is refused. Uses the run's evaluation seed, so the result equals the
read-out of the same run trained with these `residual_test` settings. The run directory is
not written to.

| Key | Meaning |
|---|---|
| `run` | Run directory (training or stream run). Required. |
| `checkpoints` | Checkpoints whose saved arrays are tested (`[best]`). |
| `sets` | Held-out sets as `split/fidelity`, e.g. `validation/hf`, `validation/stream` (stream validation events) (all held-out sets the run evaluated). |
| `residual_test` | Test settings, as in the training config (defaults). |
| `output_directory` | Output directory; must be new or empty. Required. |

## Within-design-point config

`WithinPointRunConfig`: the within-design-point metrics of the read-out (`within_point_*`, their
68/95 % bootstrap intervals, `within_point_gain_coverage`; `docs/evaluation.md`) on a finished
training or stream run, from the logits and labels it saved; for runs trained before the
read-out had them. Inputs are reloaded and checked as for the residual test, with the run's
evaluation seed, so the values equal the read-out of the same run trained with the current
toolkit. An added evaluation of the run, written into the run directory beside the read-out:
`within_point.json` (one row per checkpoint and set) and per set
`{fidelity}_{split}_{checkpoint}_within_point.npz` and `_within_point_gain.png`. The run's own
files are not changed; a run with a `within_point.json` is refused.

| Key | Meaning |
|---|---|
| `run` | Run directory (training or stream run). Required. |
| `checkpoints` | Checkpoints whose saved arrays are scored (`[best]`). |
| `sets` | Design-point sets as `split/fidelity`, e.g. `validation/lf`; no stream events (all design-point sets the run evaluated). |
| `within_point_null_draws` | Label draws per design point for the null (1000, ≥ 100), as in training configs. |
