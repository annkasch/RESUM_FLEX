# Install details

Python versions, extras, the pixi/uv split and the NumPy 1/2 matrix.

Python 3.11–3.12 is supported; `.python-version` selects 3.11 (the minimum) for local
development. PyTorch is a hard dependency. The `gp` extra installs
`GPy>=1.14.2` and `emukit>=0.5.1` for MFGP / Phase 4 & 5; `bdt` installs scikit-learn
(BDT event model); `lgbm` installs LightGBM (residual learnability test, `residual_test:`
and the `residual` stage; its tests skip without it). Test and lint tools
live in the non-published `dev` dependency group and are installed by default.

The legacy `gp-numpy1` extra pins its own NumPy and SciPy compatibility bounds
and is used by the NumPy 1 compatibility job. This split is required rather
than cosmetic: GPy 1.14.2 depends on
`paramz>=0.10`, which in turn requires NumPy 2, while GPy 1.13.2 belongs to
the older NumPy 1 stack. The extras and NumPy groups are declared mutually
exclusive, so uv rejects invalid combinations instead of silently mixing them.

The uv development lock selects the portable CPU-only PyTorch wheel. This is a
development source override only: the published package metadata remains
backend-neutral, so downstream users can select CPU, CUDA, ROCm, or another
appropriate PyTorch build.

The compatibility matrix pins Torch 2.11.0 in the development group so the
NumPy 1/2 comparison does not also change the neural-network implementation.
This exact pin is not part of the published package requirements.

## Pixi + uv: complementary ownership

`pyproject.toml` is the **only** declaration of Python package requirements,
and `uv.lock` is the **only** Python dependency lock. `pixi.toml` deliberately
contains only the `uv` executable plus task definitions; `pixi.lock` therefore
reproduces the outer tool layer without independently resolving the Python
stack.

```text
pixi.toml / pixi.lock       pyproject.toml / uv.lock
        │                              │
        └── pins and runs uv ──────────┴── resolves Python packages
```

After installing Pixi, the common commands are:

```bash
pixi run sync               # materialize .venv-numpy2 from uv.lock
pixi run test               # default NumPy 2 test suite
pixi run test-numpy1        # independent .venv-numpy1 compatibility suite
pixi run test-all           # both locked compatibility environments
pixi run numpy2-check       # NumPy 2 removed/deprecated API scan
pixi run lock-check         # fail if pyproject.toml and uv.lock diverge
```

All Pixi tasks invoke uv with `--frozen`; Pixi never installs Python packages
into its own environment. The two NumPy variants use separate virtualenvs, so
switching compatibility tests cannot mutate the other environment. The NumPy 2
test task fixes BLAS to one thread because its OpenBLAS build otherwise creates
dozens of workers for small GP matrices. The legacy NumPy 1 task fixes two
threads to preserve its existing deterministic optimization path.

## Paper aggregate replay

`notebooks/paper_2410_03873_reproduction.ipynb` replays the portions of the
RESuM paper that are supported by the aggregate artifacts in the original
repository. It covers the stored CNP diagnostics, all six recorded
active-learning updates, the three-fidelity MFGP, the 100-trial HF validation,
and the no-CNP ablation. The notebook labels aggregate replay separately from
exact recomputation and leaves one intentionally empty CNP cell because the
complete portable event corpus and current checkpoint are unavailable.

The aggregate inputs and their provenance are bundled under
`notebooks/data/paper_2410_03873/`, so the notebook does not import or read from
the original RESuM checkout:

```bash
pixi run notebook
```

The `notebook` dependency group supplies JupyterLab without publishing it as a
runtime library requirement. Pixi delegates that group to uv and materializes
it in an isolated `.venv-notebook` environment, preserving the same ownership
boundary as the test environments. The current replay reads CSV files only and
therefore does not require `h5py`.

## NumPy compatibility contract

The library declares `numpy>=1.24,<3`; uv's mutually exclusive `numpy1` and
`numpy2` groups test both sides of that range. NumPy 2 support also requires
the compatible lower bounds on compiled or NumPy-facing dependencies:

| Dependency | Declared minimum | NumPy 2 status in this project |
|---|---:|---|
| SciPy | 1.13 in `numpy2` | First SciPy release targeting NumPy 2; legacy group uses ≤1.12 |
| Matplotlib | 3.9 | Wheels are built against the NumPy 2-compatible ABI |
| PyTorch | 2.3 | NumPy bridge support; round-trip covered by a test |
| GPy | 1.14.2 | NumPy 2 compatibility release; requires paramz 0.10+ |
| Emukit | 0.5.1 | Core NumPy 2 support and current GPy integration |
| Pydantic / PyYAML | unchanged | No NumPy ABI dependency |

Run `pixi run numpy2-check` for Ruff's NumPy 2 migration rule (`NPY201`). Full
tests, including GPy fit/predict, MFGP coverage, optimizer behavior, and the
Torch↔NumPy boundary, run in both locked environments. The NumPy 1 job uses
the explicitly isolated `gp-numpy1` legacy extra.
