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

Run tests and jobs with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## Usage

```bash
.venv/bin/python -m core.surrogates train training.yaml
.venv/bin/python -m core.surrogates predict runs/mlp/checkpoints/best prepared unlabeled
.venv/bin/python -m core.surrogates mfgp gp.yaml
```

`training.yaml`:

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

`gp.yaml`:

```yaml
levels:
- {source: model_mean, partition: unlabeled, inference: runs/mlp/inference/best/unlabeled_lf.npz}
- {source: observed_fraction, partition: train, fidelity: hf}
evaluation:
- {partition: validation, fidelity: hf}
data_directory: prepared
output_directory: runs/gp
```

Test partitions are only evaluated with `unblind_test: true`.

## Your own data

By default, data are read from `<data_directory>/batches/<partition>/<fidelity>.npz`
(the arrays of a `StandardBatch`). For other formats, write a dataset provider and name it
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

See `core/surrogates/datasets.py`.

## Documentation

- [docs/configs.md](docs/configs.md): every config setting with its default
- [docs/surrogates.md](docs/surrogates.md): validation metrics, GP levels, outputs, background
- [docs/reference.md](docs/reference.md): install details, data contracts, Python API
- `scripts/phase*.py`: end-to-end demonstrations on synthetic data
