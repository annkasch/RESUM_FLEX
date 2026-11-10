# Synthetic validation, smoke check, tests

## Status

| Phase | What | State |
|---|---|---|
| 0 | Pydantic schemas (`StandardBatch`, `ModelPrediction`, `Config`) | ✅ |
| 1 | Pseudo-data generator with analytical `t(θ, φ)` + 8-scenario plots | ✅ |
| 2 | Universal encoder with learnable null embeddings (PyTorch) | ✅ |
| 3 | CNP training, checkpoints, reconstruction & coverage | ✅ |
| 4 | MFGP co-kriging via Emukit/GPy + held-out coverage gate | ✅ |
| 5 | Active learning — IVR (exploration) + EI (exploitation) | ✅ |

## Validation grid

Phases 0–5 were built and gated in May 2026 on 8 scenarios: FULL with dim(θ), dim(φ) ∈
{1,2}² (S1–S4), EVENT_ONLY with dim(φ) 1/2 (S5–S6), DESIGN_ONLY with dim(θ) 1/2 (S7–S8).
Gate per scenario: `MAE(p̂, p) < threshold` (`schemas/config.py`, `mae_thresholds`).
`data/pseudo_generator.py`, the phase scripts and the tests still use the grid. Plot rule
(`viz/dispatch.py`): 1-D line ± band, 2-D heatmap, ≥3-D slices; truth and prediction
together (overlay in 1-D, side by side with a shared colorbar in 2-D).

## Pipeline smoke check

`python scripts/smoke.py` runs every CLI stage once on tiny seeded synthetic data (S4) in a
fresh temporary directory: `train` (MLP + last layer), `predict` (three partitions), `mfgp`,
`binomial` (event-model covariate), `stream` (stream MLP on an in-memory event source,
scored on the design-point data) and, only with the `lgbm` extra, `residual_test:` in the
train and stream runs plus a `residual` stage on each; a `within_point` stage on the train run. It checks exit codes and `metrics.json`/`compute.json`
and prints the time per stage (~30 s, one thread per process, login-node safe). A pipeline
check only: no model conclusions. `--keep` keeps the directory, `--workdir DIR` runs in DIR.
`tests/test_smoke.py` runs it under the `smoke` marker (skip with `-m 'not smoke'`).

## MFGP rate bias vs. hits per point

`python scripts/mfgp_noise_sweep.py --hits 1,5,50 --transforms identity,log --seeds 0,1,2,3,4`
(~3 min, login-node safe) fits a two-level MFGP on synthetic counts with a known rate
(`core/mfgp_bias.py`): a smooth 2-D rate surface, ~20× between dim and bright points; the low
level is a slightly distorted model mean at 300 points, the top level the observed fraction
m/N at 80 points with N set for the given mean hits. Per run it prints the relative bias of
the predicted rate against the true rate (all / dim / bright held-out points), the 1σ/2σ/3σ
coverage of fresh observed fractions and, in log mode, the exact bias terms of the smoothed
log target (`log_target_terms`: pseudocount shift E[y]/p − 1, Jensen shift
exp(E[log y])/E[y] − 1) plus the back-conversion shift exp(σ²/2) − 1 of the learned top-level
noise. Rows go to `--out` (default `viz_output/mfgp_noise_sweep.json`).

Constant learned noise, 5 seeds (2026-11-09): identity mode has no systematic bias at 1, 5 or
50 hits (seed-to-seed scatter ±5–10 %); log mode overpredicts by ≈ +57 % / +21 % / +3 % at
1 / 5 / 50 hits, about pseudocount + Jensen + back-conversion (+16 − 12 + 18 % at 5 hits).
With the rate mean exp(μ + v_f/2) (noise excluded, same day): log mode ≈ +30 % / +2.5 % /
+1 % at 1 / 5 / 50 hits, the remaining pseudocount + Jensen terms; coverage unchanged.
With the log-target bias correction as well (`log_bias_max_refits`, default): ≈ +5 % /
+0.1 % / +1 % (dim points +1.5 % / −1 % / +0.5 %); 1σ coverage at 1 hit stays ≈ 0.3, as
positive lognormal bounds can't cover the many zero observed fractions there.

## Run the synthetic-validation pipelines (Phase 1–5)

For a sanity check on a fresh install, the `scripts/` directory drives
the full 8-scenario synthetic validation. All outputs go to
`viz_output/phaseN_*/` (gitignored):

```bash
python scripts/phase1_plot_ground_truth.py        # pseudo-data plots, S1..S8
python scripts/phase2_plot_latent.py              # encoder null-token PCA
python scripts/phase3_plot_reconstruction.py      # CNP train + reconstruction + coverage
python scripts/phase4_plot_mfgp.py                # MFGP posterior + coverage + Q-Q
python scripts/phase5_plot_optimizer.py           # IVR active learning + trajectory
python scripts/phase5_plot_optimizer.py \
    --acquisition ei --target min --n-initial-hf 4 \
    --scenarios S1,S8 \
    --out-dir viz_output/phase5_optimizer/stress_ei_min
```

Every Phase 5 run also emits an `optimizer_{S}_trajectory.png` and an
`optimizer_{S}_metrics.png` showing how the optimizer reaches the optimum.

## Tests

```bash
.venv/bin/python -m pytest -q                    # full suite, ~12 min on one CPU core
.venv/bin/python -m pytest -q -m 'not slow'      # quick check without the 12 slowest tests
.venv/bin/python -m pytest -q -m slow            # only the slow ones (MFGP gate, smoke, ...)
.venv/bin/python -m pytest tests/test_mfgp_recovery.py -v  # MFGP coverage gate, ~3 min
```

`slow` marks tests of 15 s or more (`pyproject.toml`); tag new ones that cross that line.
Tests of optional extras skip without them (`pytest.importorskip`): `tests/test_residual_*.py`
need `lgbm`, which the pixi NumPy 1/2 matrix does not install.
