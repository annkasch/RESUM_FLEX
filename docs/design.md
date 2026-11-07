# Design

How the code fits together: data flow, layout, where to make a change, and why the rules in
`CLAUDE.md` exist.

## Data flow

```
application DatasetProvider (core/surrogates/datasets.py; NpzBatches built in)
  └─ StandardBatch (schemas/data_models.py)  theta [B,Dθ]?, phi [B,N,Dφ]?, labels [B,N], masks
       └─ build_surrogate (core/surrogates/models.py) -> EventSurrogate (core/surrogates/base.py)
            NeuralSurrogate: cnp | mlp | field | transformer | legacy_cnp   TreeSurrogate: bdt
            fit: core/surrogates/training.py + stages.py; checkpoints.py -> checkpoints/{best,final}
            evaluate: evaluation.py (+ readout.py, residual.py) driver: experiment.run_experiment
       ├─ route A  last_layer.py (Laplace, final linear layer) -> rate draws per design point
       └─ route B  inference.py (predict: mean p per design point)
                     ├─ mfgp_stage.py -> mfgp_pipeline.py -> surrogate_mfgp.py (Emukit/GPy)
                     └─ binomial_stage.py -> binomial_gp.py (counts, optional covariate)
event store -> EventSource (core/event_stream.py) -> stream_training.py (mlp, field)
GP posterior -> core/optimizer.py (IVR / EI over θ)
```

CLI (`core/surrogates/__main__.py`): `train` -> `run_experiment`, `predict` -> inference,
`mfgp` -> `run_mfgp_config`, `binomial` -> `run_binomial_config`, `stream` -> `run_stream`,
`residual` -> `residual_stage.run_residual` (residual test on a finished run).

## Layout

```
schemas/        pydantic configs and data contracts (numpy-only)
  data_models.py  StandardBatch, ModelPrediction
  surrogates.py   run configs: event models, training, GP levels, evaluation
  config.py       typed config of the synthetic phase pipeline
core/surrogates/ shared build/fit/predict/evaluate API, run stages, CLI
core/           event models (bernoulli_*.py, surrogate_bdt.py), encoder (networks.py),
                GPs (surrogate_mfgp.py, binomial_gp.py), optimizer.py, legacy training.py
data/           synthetic data (pseudo_generator.py) and batch sources
viz/            plotting: coverage, z-scores, run summaries, projections
scripts/        smoke.py (CLI pipeline check), phase{1..5}_*.py (synthetic demonstrations)
notebooks/      minimal example, paper reproduction
tests/          pytest, synthetic data only
pyproject.toml + uv.lock   Python deps;  pixi.toml + pixi.lock   uv and tasks only
```

Module docstrings describe each file.

## Where to change what

| Change | Open first |
|---|---|
| New config field | `schemas/surrogates.py`, then the consumer; [configs.md](configs.md) |
| New event model kind | `schemas/surrogates.py` (kind), `core/surrogates/models.py`, `core/bernoulli_*.py` |
| Losses, mixup, target sampling | `core/binary_losses.py`, `core/mixup.py`, `core/target_sampling.py` |
| Training stages, checkpoint choice | `core/surrogates/stages.py`, `training.py`, `experiment.py` |
| Metrics (count NLL, coverage, CRPS) | `core/surrogates/evaluation.py`, `core/gp_metrics.py` |
| Read-out, residual learnability test | `core/surrogates/readout.py`, `core/surrogates/residual.py`, `residual_stage.py` (finished runs) |
| Bayesian last layer | `core/last_layer.py` (numpy math), `core/surrogates/last_layer.py` (torch side) |
| GP levels / fidelities | `core/surrogates/mfgp_stage.py`, `core/mfgp_pipeline.py` |
| Binomial GP | `core/surrogates/binomial_stage.py`, `core/binomial_gp.py` |
| Stream training | `core/event_stream.py`, `core/surrogates/stream_training.py` |
| Data access for applications | `core/surrogates/datasets.py` (DatasetProvider protocol) |
| Spatial projections | `core/spatial_projections.py`, `core/projection_adapters.py` |
| Plots | `viz/` (`dispatch.py:plot_coverage_test` for coverage) |
| Whole pipeline still runs? | `scripts/smoke.py` |

Legacy path, still tested: `core/surrogate_cnp.py` + `core/training.py` (two-output CNP,
synthetic phases, `legacy_cnp` kind).

## Why the rules exist

- **Null embeddings, not branches.** One encoder (`core/networks.py`) for all three modes keeps
  one code path tested. Every missing input maps to the same learnable `theta_null`/`phi_null`.
- **Aggregate over N, never B.** Latents `[B, N_ctx, Z]` -> `[B, Z]` with `mean(dim=1)`
  (`core/surrogate_cnp.py`). Averaging over B mixes unrelated design points. The loss still
  goes down, the predictions are wrong, and nothing raises an error.
- **Torch and GPy apart.** GPy/Emukit are imported only in `core/surrogate_mfgp.py`,
  `core/binomial_gp.py` and `core/optimizer.py`, and these don't import Torch. The stacks have
  different NumPy/BLAS needs and GPy is an optional extra. They exchange numpy arrays only
  (`core/mfgp_pipeline.py`, `core/surrogates/mfgp_stage.py`), so either side can be replaced
  or left out. Likewise LightGBM (optional `lgbm` extra) is imported only in
  `core/surrogates/residual.py`, inside its functions, and sees numpy arrays only.
- **Scale before batching.** The encoder MLP ignores features ≥ ~10× smaller in range. GP ARD
  lengthscales compensate, the encoder cannot. No auto-scaling: silent unit changes would
  break predict-time queries. Persist the scaler with the checkpoint ([data.md](data.md)).
- **Never clip probabilities.** A clipped p hides a broken model and biases the NLL. Count
  invalid values or fail.
- **Selection on validation, LF by default.** HF is scarce and feeds the GP; selecting on it
  would leak into the GP stage. The test set stays blinded so later comparisons are honest.
- **Coverage ≠ projected inclusion ≠ marginalized-mean uncertainty.** They answer different
  questions. Pooling them, or pooling density NLL with count-mass NLL, gives numbers that
  mean nothing.
- **BLAS threads = 1.** OpenBLAS oversubscription on small GP matrices slows runs by orders of
  magnitude.
