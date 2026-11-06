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
`binomial` (event-model covariate) and `stream` (stream MLP on an in-memory event source,
scored on the design-point data). It checks exit codes and `metrics.json`/`compute.json`
and prints the time per stage (~30 s, one thread per process, login-node safe). A pipeline
check only: no model conclusions. `--keep` keeps the directory, `--workdir DIR` runs in DIR.
`tests/test_smoke.py` runs it under the `smoke` marker (skip with `-m 'not smoke'`).

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
.venv/bin/python -m pytest -q                              # full suite, a few minutes on CPU
.venv/bin/python -m pytest -q --ignore=tests/test_mfgp_recovery.py   # without the slow GP gate
.venv/bin/python -m pytest tests/test_mfgp_recovery.py -v  # MFGP coverage gate, ~4 min
```
