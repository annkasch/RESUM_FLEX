# RESUM_FLEX

A modular refactor of [RESuM](https://openreview.net/pdf?id=lqTILjL6lP), a surrogate model
for rare-event design problems. Event models (CNP, MLP, tabular transformer, BDT) learn
per-event hit probabilities from simulated binary outcomes; a multi-fidelity Gaussian
process combines them with simulated hit fractions across fidelity levels.

## Install

```bash
git clone https://github.com/annkasch/RESUM_FLEX.git
cd RESUM_FLEX
uv sync --extra gp      # or: pixi run uv sync --extra gp
```

Optional extras: `bdt` (scikit-learn, BDT event model), `lgbm` (LightGBM, residual
learnability test in the read-out).

Run tests and jobs with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## Usage

```bash
.venv/bin/python -m core.surrogates train training.yaml
.venv/bin/python -m core.surrogates predict runs/mlp/checkpoints/best prepared unlabeled
.venv/bin/python -m core.surrogates mfgp gp.yaml
```

`training.yaml` ([all settings](docs/configs.md#training-config)):

```yaml
model: {kind: mlp, architecture: {hidden_dims: [256, 256, 256]}}
training:
  backend: neural
  loss: bernoulli
  n_passes: 10
  batch_size: 16
  n_events: 750
  sampling: {strategy: natural}
selection: bernoulli_log_loss
hf_validation: false
data_directory: prepared
output_directory: runs/mlp
```

`gp.yaml` ([all settings](docs/configs.md#gp-config), [GP levels](docs/gp.md#multi-fidelity-gp-configurable-levels)):

```yaml
levels:
- {source: model_mean, partition: unlabeled, inference: runs/mlp/inference/best/unlabeled_lf.npz}
- {source: observed_fraction, partition: train, fidelity: hf}
evaluation:
- {partition: validation, fidelity: hf}
data_directory: prepared
output_directory: runs/gp
```

As an alternative to the MFGP, `python -m core.surrogates binomial binomial.yaml` fits one GP
directly to hit counts with a binomial likelihood (no event model).

With the `lgbm` extra, `residual_test:` in a training or stream config adds the residual
learnability test to the read-out; `python -m core.surrogates residual residual.yaml` runs it
on a finished run without retraining ([docs/evaluation.md](docs/evaluation.md)).

Test partitions are only evaluated with `unblind_test: true`. What a run writes is in
[docs/runs.md](docs/runs.md), the validation metrics in [docs/evaluation.md](docs/evaluation.md).

## Your own data

By default, data are read from `<data_directory>/batches/<partition>/<fidelity>.npz`
(the arrays of a [`StandardBatch`](docs/data.md#standardbatch)). For other formats, write a dataset provider and name it
in the config with `data_provider: my_package.module:MyDataset`:

```python
class MyDataset:
    def __init__(self, root): ...
    def partitions(self): ...
    def load(self, partition, fidelity, *, labels=True): ...  # (StandardBatch, provenance)
    def theta(self, partition, fidelity): ...                 # (normalized theta, voxel ids)
    def coordinates(self, partition, fidelity): ...
    def normalization(self): ...
```

See [Datasets and providers](docs/data.md#datasets-and-providers) and
`core/surrogates/datasets.py`.

## Documentation

- [docs/index.md](docs/index.md): one file per topic (models, training, GPs, metrics, runs,
  data, math, design, install), with when to read each
- [docs/configs.md](docs/configs.md): every config setting with its default
- [docs/nersc.md](docs/nersc.md): running on NERSC Perlmutter / Slurm
- `scripts/phase*.py`: end-to-end demonstrations on synthetic data
