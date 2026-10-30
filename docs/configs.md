# Config settings

Every YAML file is validated by a pydantic model in `schemas/surrogates.py`; unknown keys
are rejected. Relative paths resolve against the YAML file. Defaults are given in
parentheses.

- [Training config](#training-config) (`python -m core.surrogates train`)
- [GP config](#gp-config) (`python -m core.surrogates mfgp`, or `mfgp:` in a training config)

## Training config

`SurrogateRunConfig`.

### Data and output

| Key | Meaning |
|---|---|
| `data_directory` | Prepared data, read by the dataset provider. Required. |
| `data_provider` | Provider as `module:Class` (none: `.npz` batches, `batches/<partition>/<fidelity>.npz`). |
| `data_config` | Data-preparation YAML for the provider; the data are rebuilt when it or the raw files change (none). |
| `partitions` | Provider partitions for `train`, `validation`, `test` (each named like the key). |
| `output_directory` | Run directory. Required; must be new or empty. |

### Evaluation

| Key | Meaning |
|---|---|
| `selection` | Checkpoint metric on LF validation: `bernoulli_log_loss`, `average_precision`, `voxel_rate_mae` (`voxel_rate_mae`). Use log loss for natural-event runs, AP for mixup runs. |
| `lf_validation` | Select on LF validation; `false` keeps the final checkpoint (`true`). |
| `hf_validation` | Also report HF validation (`true`). |
| `test_fidelities` | Test sets that exist, e.g. `[lf]` (`[]`). |
| `unblind_test` | Evaluate the test sets. Leave `false` until all choices are final (`false`). |
| `validation_context_events` | Context events per voxel for context models during evaluation (64). |
| `validation_context_sweep` | Extra context sizes evaluated after training, on the same targets (`[]`). |
| `validation_seed` | Seed of the evaluation context/target split (12345). |
| `selection_max_voxels` | Select checkpoints on a seeded subset of this many validation voxels (all). |
| `evaluation_threads` | CPU threads for evaluation passes (1). |

### Model (`model`)

`kind` selects the model; the remaining keys depend on it.

| `kind` | Keys |
|---|---|
| `cnp` | `encoder`: `latent_dim`, `hidden_dims`, `dropout` (64, `[128, 128]`, 0). Uses context events. |
| `mlp` | `architecture`: `hidden_dims`, `dropout` (`[128, 128]`, 0). |
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
