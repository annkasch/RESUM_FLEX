# Evaluation

Event-model validation metrics, read-out after training and blinded test sets.

## Validation metrics

The event models output per-event hit probabilities, so they are judged as
probabilities: right order (discrimination) and right level (calibration).
`evaluate_surrogate` reports, for one episode (targets only, padding excluded):

| Metric | Level | Meaning | Ideal |
|---|---|---|---|
| `bernoulli_log_loss` | event | proper score; recommended `selection` for natural-event runs | low |
| `log_loss_skill` | event | `1 - LL / LL_const`, constant = training prevalence | > 0 |
| `brier_score` | event | mean `(y - p)^2` | low |
| `average_precision`, `lift` | event | order of events only; lift = AP / prevalence | lift > 1 |
| `calibration_z` | all | `(Σk - Σp) / sqrt(Σp(1-p))`, overall level | \|Z\| < 3 |
| `calibration_chi2_per_bin` | event bins | 10 equal-count bins of p, mean `(K-M)^2/V` | ≈ 1 |
| `voxel_z_mean`, `voxel_z2_mean` | voxel | `z = (k - Σp) / sqrt(Σp(1-p))` | 0, 1 |
| `voxel_rate_mse_denoised` | voxel | mean of `(o - q)^2 - o(1-o)/(n-1)` | ≈ 0 |
| `spearman_r` | voxel | rank correlation of `o` and `q` (trend) | near ceiling |

Per voxel: `o` = hits / target events, `q` = mean predicted p over the same
targets, `n` = target events. The subtracted term is the unbiased binomial noise
estimate, so `voxel_rate_mse_denoised` estimates the real `(t - q)^2`; it can be
slightly negative and is never clipped. `voxel_rate_mae` is kept for old runs
only: with few hits per voxel it is dominated by noise and favours the median
count (it prefers 0 in voxels with < 0.7 expected hits). AP needs natural
probabilities only for its baseline; use it for mixup ablations, whose outputs are
scores, not probabilities. The log-loss skill needs the training prevalence and is
`None` without it.

Evaluation layout: each voxel's events are permuted with `validation_seed`. The
targets are the events after `validation_context_pool = max(validation_context_events,
*validation_context_sweep)`; the context is the first `n` of the pool. Every context
size therefore scores the same targets, and contexts are nested.

Read-out after training (held-out splits only, not used for selection):

- `spearman_noise_ceiling`: median Spearman r over 200 label sets redrawn from the
  model's own p, i.e. what a model equal to the truth would reach at this noise.
- `*_ci95`: 95% bootstrap intervals over voxels (200 resamples) for log loss, AP and
  Spearman r. Use them to compare runs.
- `context_sweep.json` / `context_sweep.png` (context models only): the metrics at
  every size in `validation_context_events` + `validation_context_sweep`, plus the
  standard size with all context labels set to 0. A flat curve, or zeroed labels
  scoring like observed ones, means the model ignores its context labels.
- `{fidelity}_calibration.png`: observed hit fraction vs mean p per bin, Wilson 95%.

Seeds: `SeedSequence([validation_seed, k])`, k = 0 selection subset, 1 noise
ceiling, 2 bootstrap.

## Blinded test sets

Test partitions are blinded by default. `test_fidelities` declares which test sets exist,
but a training run evaluates them only with `unblind_test: true`; otherwise it writes
`test_evaluation.json` with `status: blinded` and never reads the test data. The MFGP
does the same: evaluation on `test`, `cnp_test` or `gp_test` needs `unblind_test: true`,
and test partitions can never be GP training levels. Make all model and GP choices on
validation, then unblind once.
