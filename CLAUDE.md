# RESUM_FLEX — agent brief

Application-independent toolkit for rare-event surrogates, a modular refactor of RESuM
([paper](https://openreview.net/pdf?id=lqTILjL6lP)). Event models learn per-event outcome
probabilities from simulated binary outcomes; a Bayesian last layer or a Gaussian process turns
them into rates with uncertainty per design point. No application code lives here: applications
bring their data through a dataset provider and keep their own configs, outputs and job scripts.
Keep the vocabulary generic (design point, event, outcome), also in code names; no application
terms such as voxels.

If `.claude/status.md` exists (local, open items only), read it first. On a Slurm cluster (e.g.
NERSC Perlmutter), read `docs/nersc.md` first: login nodes take one single-threaded process of up
to ~20 min.

Docs: `README.md` (usage); `docs/index.md` lists one file per topic and when to read it.
Open only the file the task needs: `docs/design.md` before changing code (data flow, which
file to open, why the rules exist), `docs/math.md` for losses and GP math, `docs/configs.md`
for config settings.

## Concepts
- `θ` design parameters (one per simulation trial / design point), `φ` event parameters,
  `X` binary outcomes. Modes FULL (θ,φ,X), EVENT_ONLY (φ,X), DESIGN_ONLY (θ,X): missing inputs
  use learnable null embeddings (`core/networks.py`), never separate code paths.
- Two ways from event probabilities to design-point rates, compared on the same predictive
  metrics (count NLL, coverage, CRPS):
  - Event model + Bayesian last layer (`model.last_layer`, Laplace over the final linear
    layer; mlp, cnp, field, transformer): rates with uncertainty directly, no GP. The
    approach under test.
  - Event model → mean p per design point (`predict`) → GP over configurable fidelity levels
    (`mfgp`), or a binomial count GP directly on outcome counts (`binomial`, optional
    event-model covariate). The baseline.
- Event model kinds (`schemas/surrogates.py`): `cnp`, `mlp`, `field` (neural field for
  point-cloud data, 3-D direction), `transformer`, `bdt`, `legacy_cnp`. `mlp`, `field`, `transformer` and `bdt`
  also train on an event stream (`stream`; BDT on an in-memory case-control sample). No stream
  CNP: ungrouped events have no context sets (`docs/stream.md`).
- Optional: IVR/EI active learning over θ (`core/optimizer.py`, GP-based); residual
  learnability test in the read-out (`residual_test:`, `core/surrogates/residual.py`, LightGBM).
- Finished runs are re-scored from their saved logits without retraining: `residual`,
  `within_point` (`core/surrogates/finished_run.py` reloads and checks the inputs). `score`
  predicts prepared sets anew with a finished last-layer model and scores them like the
  binomial GP; donor event inputs only with the provider's `transfer_events` rule.

## Environment
- Repo-local `.venv` from `pyproject.toml` + `uv.lock`; call `.venv/bin/python`,
  `.venv/bin/pytest`, `.venv/bin/ruff` directly. Sync: `uv sync --extra gp --extra bdt --extra
  lgbm` (`gp` = GPy + Emukit, `bdt` = scikit-learn, `lgbm` = LightGBM; `gp-numpy1` is the legacy
  NumPy-1 matrix only and conflicts with `gp`).
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
.venv/bin/pytest -q                          # full suite, ~12 min on one core
.venv/bin/pytest -q -m 'not slow'             # without the 12 slowest tests, ~5 min
.venv/bin/ruff check .                       # repo-wide; must stay clean (notebooks excluded)
.venv/bin/ruff format <files>                # format only files you touched
.venv/bin/python -m core.surrogates train TRAINING.yaml
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION [--fidelity hf] [--event-logits]
.venv/bin/python -m core.surrogates mfgp|binomial|stream|residual|within_point|score CONFIG.yaml
.venv/bin/python scripts/smoke.py    # every CLI stage on tiny synthetic data, ~30 s
.venv/bin/python scripts/phase1_plot_ground_truth.py   # cheap synthetic sanity check
.venv/bin/python scripts/mfgp_noise_sweep.py           # MFGP rate bias vs. hits per point, ~1 min
```
After a change, run `scripts/smoke.py` (pipeline check only; also `pytest -m smoke`, skip with
`-m 'not smoke'`).

## Layout
- `schemas/`: pydantic configs and data contracts (numpy-only); all run configs in `surrogates.py`.
- `core/surrogates/`: shared build/fit/predict/evaluate API, run stages and the CLI (`__main__.py`);
  entry point `experiment.run_experiment`, data access via `datasets.py` (DatasetProvider).
- `core/`: models and math (`core/quantile_residuals.py`: randomized quantile residuals of hit
  counts behind every z-score plot; `core/provenance.py`: the `code` block, resum_flex commit,
  dirty flag, package versions, command line and the config-loaded application modules, in
  every run's JSON, and the YAML copy); `data/`: synthetic data; `viz/`: plotting
  (`viz/dispatch.py:plot_coverage_test` is the canonical coverage renderer).
- `scripts/smoke.py` (CLI pipeline check), `scripts/phase{1..5}_*.py`,
  `scripts/mfgp_noise_sweep.py` (MFGP bias on synthetic counts, `core/mfgp_bias.py`), `notebooks/`,
  `tests/` (synthetic data only).
  Module docstrings and `docs/design.md` describe the files.

## Conventions
- Config: every YAML loads into a pydantic model in `schemas/`; change behaviour through YAML.
  No per-model scripts, no inert config fields. Paths resolve relative to the YAML file.
- Run outputs: one `output_directory` per run, never overwritten. A run writes `experiment.json`
  (resolved config, input hashes, normalization), a verbatim copy of its YAML (`<stage>.yaml`), `checkpoints/{best,final}/`, `history.json`,
  `metrics.json`, `compute.json`, evaluation arrays and plots (see `docs/runs.md`).
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
  LightGBM is imported only in `core/surrogates/residual.py` (numpy in and out).
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
this file in the same commit. Local notes (`.claude/`, incl. `status.md`) are never committed.
`status.md` holds open items only (now, waiting on the user, next, known issues), one line per
item: at the end of each task delete finished items and add new ones; no history.
