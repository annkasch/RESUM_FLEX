# RESUM_FLEX — agent brief

Application-independent toolkit for rare-event surrogates, a modular refactor of RESuM
([paper](https://openreview.net/pdf?id=lqTILjL6lP)). Event models learn per-event outcome
probabilities from simulated binary outcomes; a Bayesian last layer or a Gaussian process turns
them into rates with uncertainty per design point. No application code lives here: applications
bring their data through a dataset provider and keep their own configs, outputs and job scripts.
Keep the vocabulary generic (design point, event, outcome); application terms such as voxels or
photons appear only in code names.

If `docs/PROGRESS.md` exists (local notes), read it first. On a Slurm cluster (e.g. NERSC
Perlmutter), read `docs/nersc.md` before running anything heavier than a single test file.

Docs: `README.md` (usage), `docs/configs.md` (every config setting),
`docs/surrogates.md` (metrics, GP levels, run outputs, model kinds),
`docs/reference.md` (install, layout, data contracts, Python API, synthetic phases).

## Concepts
- `θ` design parameters (one per simulation trial / design point), `φ` event parameters,
  `X` binary outcomes. Modes FULL (θ,φ,X), EVENT_ONLY (φ,X), DESIGN_ONLY (θ,X): missing inputs
  use learnable null embeddings (`core/networks.py`), never separate code paths.
- Two ways from event probabilities to design-point rates, compared on the same predictive
  metrics (count NLL, coverage, CRPS):
  - Event model + Bayesian last layer (`model.last_layer`, Laplace over the final linear
    layer; mlp, cnp, field): rates with uncertainty directly, no GP. The approach under test.
  - Event model → mean p per design point (`predict`) → GP over configurable fidelity levels
    (`mfgp`), or a binomial count GP directly on outcome counts (`binomial`, optional
    event-model covariate). The baseline.
- Event model kinds (`schemas/surrogates.py`): `cnp`, `mlp`, `field` (neural field on encoded
  coordinates; trainable on ungrouped events via `photon`), `transformer`, `bdt`, `legacy_cnp`.
- Optional: IVR/EI active learning over θ (`core/optimizer.py`, GP-based).

## Environment
- Repo-local `.venv` from `pyproject.toml` + `uv.lock`; call `.venv/bin/python`,
  `.venv/bin/pytest`, `.venv/bin/ruff` directly. Sync: `uv sync --extra gp --extra bdt`
  (`gp` = GPy + Emukit, `bdt` = scikit-learn; `gp-numpy1` is the legacy NumPy-1 matrix only and
  conflicts with `gp`).
- `pyproject.toml` is the only dependency declaration; pixi only provides uv and tasks
  (`pixi run test-all` = NumPy 1 + 2 matrix in `.venv-numpy1/2`). Never edit `uv.lock` or
  `pixi.lock` by hand.
- The Torch lock is CPU-only. CUDA is opt-in via `training.device: cuda` and needs a CUDA
  Torch from elsewhere (e.g. a cluster module).
- Always set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for tests and runs
  (OpenBLAS oversubscribes on small GP matrices).

## Commands
```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/pytest -q tests/test_schemas.py    # single files: fast
.venv/bin/pytest -q                          # full suite: heavy, not on shared login nodes
.venv/bin/ruff check <files>                 # repo-wide check has pre-existing errors
.venv/bin/ruff format <files>                # format only files you touched
.venv/bin/python -m core.surrogates train TRAINING.yaml
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION [--fidelity hf]
.venv/bin/python -m core.surrogates mfgp|binomial|photon CONFIG.yaml
.venv/bin/python scripts/phase1_plot_ground_truth.py   # cheap synthetic sanity check
```
No committed smoke config; smoke runs use a temporary config with a few steps.

## Layout
- `schemas/`: pydantic configs and data contracts (numpy-only); all run configs in `surrogates.py`.
- `core/surrogates/`: shared build/fit/predict/evaluate API, run stages and the CLI (`__main__.py`);
  entry point `experiment.run_experiment`, data access via `datasets.py` (DatasetProvider).
- `core/`: models and math; `data/`: synthetic data; `viz/`: plotting
  (`viz/dispatch.py:plot_coverage_test` is the canonical coverage renderer).
- `scripts/phase{1..5}_*.py`, `notebooks/`, `tests/` (synthetic data only).
  Module docstrings and `docs/reference.md` describe the files.

## Conventions
- Config: every YAML loads into a pydantic model in `schemas/`; change behaviour through YAML.
  No per-model scripts, no inert config fields. Paths resolve relative to the YAML file.
- Run outputs: one `output_directory` per run, never overwritten. A run writes `experiment.json`
  (resolved config, input hashes, normalization), `checkpoints/{best,final}/`, `history.json`,
  `metrics.json`, `compute.json`, evaluation arrays and plots (see `docs/surrogates.md`).
- Seeding: explicit `seed` fields; derive streams with
  `np.random.default_rng(np.random.SeedSequence([seed, k]))` and `torch.manual_seed`. No unseeded
  RNG. Only exception: GPy needs `np.random.seed` (`mfgp_stage.py`, `binomial_gp.py`).
- Statistics: normalization and GP hyperparameters from training partitions only. Checkpoint
  selection on validation, LF by default (`selection_fidelity`); never on test. Test sets are
  blinded unless `unblind_test: true`. Never clip invalid probabilities; count them or fail.
  Keep coverage, projected inclusion and marginalized-mean uncertainty distinct; don't pool
  continuous-density and count-mass NLL.
- Input scaling: the encoder MLP is scale-blind for features differing ≥ ~10× in range.
  Normalize with `core.scaling.MinMaxScaler` before building a `StandardBatch` and persist the
  scaler; `ScaleImbalanceWarning` only detects, never auto-scales.
- Decoupling: PyTorch and GPy/Emukit stacks stay separate. GPy/Emukit are imported only in
  `core/surrogate_mfgp.py`, `core/binomial_gp.py`, `core/optimizer.py`; they meet the Torch
  models only through numpy (`core/mfgp_pipeline.py`, `core/surrogates/mfgp_stage.py`).
  `core/networks.py` must not import `core/surrogate_mfgp.py`; `schemas/` stays numpy-only.
- CNP aggregator reduces over the event axis N, never the batch axis B.
- Tensor shapes as inline comments, e.g. `z_phi: torch.Tensor,  # [B, N, Z]`
  (B = design points/trials, N = events, Z = latent).
- Style: ruff, line length 100, rules E,F,I,B,UP, py311. Type hints, pydantic validation with
  informative errors, a pytest with synthetic fixtures for every change.
- Working style: check the code rather than guess; ask when intent or data format is unclear;
  push back when a request looks wrong.

## Don't
- Don't read: `viz_output/`, `build/`, `.venv*/`, lock files, notebook output cells,
  `notebooks/data/**/*.csv`; in run directories, read the JSON files (`metrics.json`,
  `compute.json`, `experiment.json`), not arrays, checkpoints or plots
  (`*.npz/*.ckpt/*.pt/*.pkl/*.png`).
- Don't edit: lock files by hand; `notebooks/data/paper_2410_03873/` (paper provenance).
- Don't commit generated outputs, application data or configs, or scratch notebooks/scripts.
- Don't put application-specific code, paths or configs into this repository.

## Workflow
After each finished, tested step: commit the code, tests and docs you changed (stage files
explicitly, leave the user's uncommitted edits out) and push the current branch to `origin`
(the fork), never `upstream`. When you add, rename or remove modules, commands or rules, update
this file in the same commit. Local notes (`docs/PROGRESS.md`, `.claude/`) are never committed;
update PROGRESS at the end of each task.
