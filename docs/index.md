# Docs index

One topic per file. Read the one you need, not all of them.

| File | Read when |
|---|---|
| [../README.md](../README.md) | First use: install, the three CLI steps, minimal configs |
| [configs.md](configs.md) | Looking up any config setting and its default |
| [design.md](design.md) | Changing code: data flow, layout, which file to open, why the rules exist |
| [math.md](math.md) | Losses, rate estimates, GP recursion, IVR, calibration target |
| [data.md](data.md) | `StandardBatch`, input scaling, dataset providers, unequal event counts |
| [models.md](models.md) | Event model kinds, the shared Python API, neural field, legacy CNP |
| [training.md](training.md) | Loss, sampling, mixup, episode building, real-plus-mixup objective |
| [stream.md](stream.md) | Training the MLP, field, transformer or BDT on an ungrouped event stream |
| [last-layer.md](last-layer.md) | Bayesian last layer: rates with uncertainty without a GP |
| [runs.md](runs.md) | CLI stages, run outputs, compute accounting, `predict` |
| [evaluation.md](evaluation.md) | Event-model metrics, read-out, blinded test sets |
| [gp.md](gp.md) | MFGP levels, GP stage after training, log-space MFGP, GP metrics, binomial GP |
| [projections.md](projections.md) | Spatial projections of a fitted GP |
| [gp-api.md](gp-api.md) | MFGP and active learning from Python |
| [python-api.md](python-api.md) | Configs in code, legacy CNP training and checkpoints |
| [install.md](install.md) | Python versions, extras, pixi/uv, NumPy 1/2 matrix, paper replay notebook |
| [synthetic.md](synthetic.md) | Phase scripts, validation grid, smoke check, tests |
| [nersc.md](nersc.md) | Running on NERSC Perlmutter / Slurm |

Paper (problem setup, CNP loss, MFGP / IVR): [openreview lqTILjL6lP](https://openreview.net/pdf?id=lqTILjL6lP).
