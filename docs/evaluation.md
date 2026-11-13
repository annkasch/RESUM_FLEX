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
| `design_point_z_mean`, `design_point_z2_mean` | design point | `z = (k - Σp) / sqrt(Σp(1-p))` | 0, 1 |
| `design_point_rate_mse_denoised` | design point | mean of `(o - q)^2 - o(1-o)/(n-1)` | ≈ 0 |
| `spearman_r` | design point | rank correlation of `o` and `q` (trend) | near ceiling |
| `within_point_log_loss_gain` | within design point | log-loss gain over each design point's own mean p, nats per positive | > 0 |
| `within_point_log_loss_skill` | within design point | same gain as `1 - LL / LL(p̄_b)` | > 0 |
| `within_point_auc` | within design point | pair AUC, hit vs non-hit of the same design point | > 0.5 |
| `within_point_spread_z` | within design point | `(ΣG - ΣE) / sqrt(ΣV)`: observed vs model-expected gain | \|Z\| < 3 |

Per design point: `o` = hits / target events, `q` = mean predicted p over the same
targets, `n` = target events. The subtracted term is the unbiased binomial noise
estimate, so `design_point_rate_mse_denoised` estimates the real `(t - q)^2`; it can be
slightly negative and is never clipped. `design_point_rate_mae` is kept for old runs
only: with few hits per design point it is dominated by noise and favours the median
count (it prefers 0 in design points with < 0.7 expected hits). AP needs natural
probabilities only for its baseline; use it for mixup ablations, whose outputs are
scores, not probabilities. The log-loss skill needs the training prevalence and is
`None` without it.

### Dependence on the event inputs (within design point)

All metrics above are dominated by the rate of each design point: a model that predicts
one constant p per design point scores almost as well as one that gets the φ-dependence
right. The `within_point_*` metrics compare events of the same design point only, so the
rate cancels. The baseline gives every event of design point b the model's own mean
`p̄_b` (not the observed rate, which is fitted to the scored labels and mixes the rate
error back in). Per event the gain is `loss(p̄_b) - loss(p_i) = a_i y_i + c_i` with
`a_i = logit p_i - logit p̄_b`, `c_i = log(1-p_i) - log(1-p̄_b)`; per design point
`G_b = Σ_i (a_i y_i + c_i)`. A constant-per-design-point model scores exactly 0 (gain) and
0.5 (AUC).

- The gain is reported per positive (`ΣG / Σk`): per event it scales with the prevalence;
  per positive it does not (e.g. `p ∝ 0.5 + φ`, φ uniform: ≈ 1/24 nats per hit at any rate).
- AUC ranks only; the gain also judges the size of the spread. A good AUC with a negative
  gain means the order is right but the spread too wide.
- If labels followed the model's p, `G_b` would have mean `E_b = Σ KL(p_i ‖ p̄_b)` and
  variance `V_b = Σ p_i(1-p_i) a_i²`. `within_point_spread_z < 0`: spread too wide or wrong
  (overconfident in φ); `> 0`: true φ-dependence stronger than predicted.
- They need grouped data (design points). Unbinned streams are not binned for them: too
  few positives per cell. Stream models get them on design-point evaluation sets.
- Finished runs trained before these metrics existed: `python -m core.surrogates
  within_point WITHIN_POINT.yaml` (`docs/configs.md`) computes them, the intervals and the
  null coverage from the saved logits and labels, without retraining, and writes them into
  the run directory (`within_point.json`).

### Calibration per input feature

`feature_calibration` (read-out and stream validation; bin count from `feature_calibration:
{bins: 10}` in the config): for each input column, equal-count
bins of its value pooled over all events and design points; per bin hits vs Σp, `z = (K -
M) / sqrt(V)`. Reported per column as `chi2_per_bin` (≈ 1) and `max_abs_z`. Columns are
`theta_j` (repeated over the events of a design point) and `phi_j`, or `record_j` for
stream records. It says which input is wrong and in which direction; it sees projections
only, so errors that cancel along a column (interactions) are missed, and if the φ
distribution differs between design points, rate differences can partly mimic or hide a
φ-trend (the `within_point_*` metrics do not have this problem).

### Residual learnability test

`residual_test` (read-out, opt-in via `residual_test:` in the config, `lgbm` extra; module
`core/surrogates/residual.py`). Feature calibration sees projections only; the
within-design-point gain says how much φ-dependence the model has, not what it misses. This
test asks whether anything learnable is left: LightGBM (`binary` objective = BCE,
`init_score` = model logit) learns `logit p' = logit p + f(θ, φ, logit p)`.

- Cross-fitting: design points go, whole, into `folds` folds (seeded, balanced by positive
  count); the trees of fold f are fitted on the other folds and score fold f, so every
  event is scored once by trees that never saw its design point (no leakage through the
  shared θ or the shared context labels). Stream validation records are independent and are
  split by event; with `max_positives` (default 20 000) only a seeded uniform fraction of
  them is used, so that about that many positives remain (precision is set by positives,
  cost by events; e.g. 20 M events at a ratio 0.003 → 6.7 M). Stream results add
  `events_available`, `positives_available`, `subsample_fraction`.
- Trees: small fixed trees; per outer fold the number of trees comes from early stopping on
  one inner fold of the training folds (same split rule), then the trees are refitted on
  all training folds. Zero trees (p' = p) if no tree beats p on the inner fold.
- `gain = (L(p) - L(p')) / Σk`, nats per positive, `L` = summed BCE. If p is right its
  expectation is ≤ 0 (p is the minimizer and the trees are fitted elsewhere), so a gain whose
  `gain_ci95` excludes 0 means p misses structure in (θ, φ).
- Design-point data split it exactly: `phi_gain = (W(p') - W(p)) / Σk` with `W` the gain
  over each design point's own mean (as `within_point_log_loss_gain`), `rate_gain = gain -
  phi_gain`. Streams report the total only.
- `*_ci68`, `*_ci95`: bootstrap over design points (streams: 1000 seeded random event
  blocks), 200 resamples. `trees` per fold, `folds`, `positives`; all gains `None` with a `skipped`
  reason below `min_positives` or with fewer than two units.
- Arrays: `residual_logits` (logit p'; padding and unused stream events NaN),
  `residual_fold` (−1 there), per design point `residual_gain_per_point`,
  `residual_phi_gain_per_point` (nats).
- Correction plot `{fidelity}_{split}_residual_correction.png` (stream:
  `stream_residual_correction.png`; best checkpoint): what the trees changed. Equal-count bins
  along logit p and along each input column (bin count from `feature_calibration.bins`),
  per bin the trees' rate change log(Σp'/Σp) over the observed log(hits/Σp) with the central
  68 % Poisson interval of the hits (bins without hits: upper limit). If the trees follow the
  observed points, the gain sits along that column; streams bin the scored (kept) events.
  Arrays `residual_correction_{events,hits,predicted,corrected,value}` [1+F, bins] (Σp,
  Σp', median column value) and `residual_correction_names` (`logit_p` first). Grouped and
  stream data alike; not written when the test is skipped.
- Cost: 2 LightGBM fits per fold, ≈ 1.5 s for 6·10⁴ events on one thread.
- Finished runs: `python -m core.surrogates residual RESIDUAL.yaml` (`docs/configs.md`)
  runs the same test on the saved logits and labels, without retraining; it writes
  `residual_test.json`, `{fidelity}_{split}_{checkpoint}_residual.npz` and
  `..._residual_correction.png` into its own output directory (bins from the run's
  `feature_calibration`).

Precision is set by positives, not design points. A positive gain says that something is
learnable, not what; the correction plot and the per-feature calibration locate it.

Evaluation layout: context-free models (mlp, transformer, field, bdt) score every event
of every design point, as stream runs do. For context models (cnp, legacy_cnp) each
design point's events are permuted with `validation_seed`; the targets are the events after
`validation_context_pool = max(validation_context_events, *validation_context_sweep)` and
the context is the first `n` of the pool. Every context size therefore scores the same
targets, and contexts are nested. `experiment.json` records the layout as
`evaluation_targets` (`all_events` or `after_context_pool`). Runs from before 2026-11-09
lack the key and scored context-free models after the context pool too, so their metrics
are not comparable with newer runs.

Read-out after training (held-out splits only, not used for selection):

- `spearman_noise_ceiling`: median Spearman r over 200 label sets redrawn from the
  model's own p, i.e. what a model equal to the truth would reach at this noise.
- `*_ci68`, `*_ci95`: central 68 % (±1σ) and 95 % bootstrap intervals over design points, from the
  same 200 resamples, for log loss, AP, Spearman r, `within_point_log_loss_gain` and
  `within_point_auc`. Use them to compare runs.
- `within_point_gain_coverage`: labels redrawn `within_point_null_draws` times (1000) from
  the model's own p give the null distribution of each `G_b`; fraction of design points
  inside its ±1/2/3σ quantiles (Gaussian tail probabilities; expected 68 / 95 / 99.7 %).
  Design points with constant predictions are skipped (`None` if all are). A band edge
  rests on the draws beyond it: at 1000 draws ~160 (1σ), ~23 (2σ), but only ~1.3 (3σ), so
  the 3σ coverage is rough; 10 000 draws give ~13. Cost ∝ draws, ≈ 10 s per 10⁶ events at
  1000. Arrays `within_point_gain_null`
  [3, 2, B]; plot `{fidelity}_{split}_within_point_gain.png` (best checkpoint).
- `feature_calibration` (above); plot `{fidelity}_{split}_feature_calibration.png`.
- `design_point_quantile_z` (array, no metric): randomized quantile residual of each design
  point's hits (`core/quantile_residuals.py`). With F the predictive CDF of the hits,
  u ~ U(F(k−1), F(k)) and z = Φ⁻¹(u); exactly N(0, 1) for a right model at any expected
  count. The Pearson z of `design_point_z_mean` is discrete and skewed when Σp per point is ~1
  (its mean and z² stay unbiased), so the z-score plot `{fidelity}_{split}_zscore.png`
  uses the quantile residual. F: Poisson-binomial of the event p, exact for k ≤ 200, normal
  with continuity correction above. Seeded by `validation_seed` (key 4); a count with zero
  predicted mass gives ±inf, counted in the plot. The GP stages save the same residual as
  `quantile_z` in `evaluation_<e>.npz` ([gp.md](gp.md)).
- `residual_test` (above), only with `residual_test:` in the config.
- `context_sweep.json` / `context_sweep.png` (context models only): the metrics at
  every size in `validation_context_events` + `validation_context_sweep`, plus the
  standard size with all context labels set to 0. A flat curve, or zeroed labels
  scoring like observed ones, means the model ignores its context labels.
- `{fidelity}_calibration.png`: observed hit fraction vs mean p per bin, Wilson 95%.

Seeds: `SeedSequence([validation_seed, k])`, k = 0 selection subset, 1 noise
ceiling, 2 bootstrap, 3 within-design-point null, 4 residual test (sub-keys 0 folds,
1 bootstrap, 2 tree seeds, 3 stream blocks, 4 stream subsample). Stream runs use `evaluation_seed`.

## Blinded test sets

Test partitions are blinded by default. `test_fidelities` declares which test sets exist,
but a training run evaluates them only with `unblind_test: true`; otherwise it writes
`test_evaluation.json` with `status: blinded` and never reads the test data. The MFGP
does the same: evaluation on `test`, `cnp_test` or `gp_test` needs `unblind_test: true`,
and test partitions can never be GP training levels. Make all model and GP choices on
validation, then unblind once.
