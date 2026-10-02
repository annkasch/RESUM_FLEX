# Statistical review: CNP denoising → MFGP (2026-10-02)

Review of the maths and statistics of the optical CNP → 3-level MFGP pipeline against the code.
No code was changed during the review. Tick items off here and record outcomes in
`docs/PROGRESS.md` (Experiment log / Decisions).

## Why the CNP step works (reference)
- Events in voxel θ: `X_i ~ Bernoulli(t(θ, φ_i))`. Target: `t̄(θ) = E_φ[t(θ, φ)]`.
- `m/N` is unbiased with `Var = t̄(1−t̄)/N`. Replacing each `X_i` by `β_i = E[X_i | φ_i] = t(θ, φ_i)`
  (conditional Monte Carlo, i.e. Rao–Blackwellization) keeps the estimate unbiased and gives
  `Var(β̄) = Var_φ(t)/N ≤ t̄(1−t̄)/N`. Law of total variance: `Var(X) = E[t(1−t)] + Var_φ(t)`;
  averaging β removes the first (coin-flip) term.
- Rare-event regime: `Var(β̄)/Var(m/N) ≲ t_max` (the largest per-event hit probability), so the
  gain is large. If φ fully determined the outcome there would be no gain.
- The argument holds only if β is (a) **calibrated**: `E[X | φ] = β`, and (b) **not fitted to
  the labels it averages**. Otherwise CNP error becomes bias, or the noise leaks back in.
- MFGP link: `y_raw − β̄_HF` should be pure binomial noise. So the top level should learn
  `ρ ≈ 1` and `δ ≈ 0`, with β̄_HF acting as a control variate for `y_raw`.

## Findings to fix (most severe first)

- [ ] **1. Mixup miscalibrates β (high).** `core/mixup.py` (`ClassAwareMixupSource.next`): in
  pools containing a positive, every negative anchor gets a positive partner with soft label
  `w ~ Beta(α, α)` (mean 0.5). BCE then fits β to roughly "0.5 × P(pool has a positive)", not to
  `t`. β̄ is a monotone score, not a probability. `mix_context: true` also trains the context
  encoder on non-physical contexts.
  *Fix:* before the MFGP, use natural-event BCE (last-layer fine-tuning or plain natural
  sampling). Test with `experiments/optical_mfgp_comparison.yaml`: expect ρ → 1 and lower bias.
- [ ] **2. In-sample LF CNP means (high).** `core/surrogates/mfgp_stage.py` builds `Y_lf_cnp` from
  the same train/LF voxels the CNP was trained on, so memorized `m/N` noise leaks into level 0.
  HF voxels are out-of-sample, so the levels are treated inconsistently.
  *Fix:* K-fold cross-fitting of the CNP over LF voxels.
- [ ] **3. Constant GP noise vs heteroscedastic binomial noise (high in log mode).**
  `Var(m/N) = p(1−p)/N`; in log space it is about `(1−p)/(Np)` (log-sd ≈ 0.32 at Np = 10, ≈ 0.63 at
  Np = 2.5). One noise variance per level overstates the noise for bright voxels and understates
  it for dim ones.
  *Fix:* fixed per-point HF noise via the delta method (GPy heteroscedastic Gaussian), or judge
  the MFGP against the binomial count GP on matched targets.
- [ ] **4. Zero-hit exclusion biases the raw fractions (medium–high).** `E[m/N | m>0] = p/(1−(1−p)^N)`:
  ×1.58 at Np = 1, ×1.05 at Np = 3. Validation is filtered the same way, so coverage applies
  only to the bright subset.
  *Fix:* include the `zero` folders in training and validation. Uncommitted edits to
  `config.optical.data.yaml` add them; check which prepared dataset the MFGP run actually uses.
- [ ] **5. `cnp_log_floor: 1e-12` (medium).** `log(1e-12) ≈ −27.6` versus typical values of about −6;
  any floored row is an extreme outlier.
  *Fix:* confirm `floored_cnp_rows == 0` in each run's `mfgp/transform.json`, and use a floor
  tied to the data (e.g. 0.5/N_target).
- [ ] **6. CNP "epistemic" sigma is constant (medium, diagnostic only).** In `core/surrogate_cnp.py`
  (`resum_binary_moments`), `scale ≥ softplus(0) ≈ 0.693` regardless of the output, so
  `cnp_trial_predictive`'s `sigma_epistemic ≈ 0.69/√N_t`, drawn from a Normal outside [0, 1]. BCE
  constrains only p̄, so σ is not identifiable. The MFGP rightly ignores it.
  *Fix:* stop reporting it as uncertainty, or drop/relabel it.
- [ ] **7. Coverage has low statistical power (medium).** With 46 HF validation voxels the
  standard error of the 68% fraction is about 0.07 (more with spatial correlation); 8/10 has an
  error of about 0.15. LF coverage (level 0 fitted to CNP means vs raw fractions) is not a
  coverage test.
  *Fix:* report interval score and CRPS alongside coverage (already computed in
  `core/gp_metrics.py`), and add uncertainty on the coverage fractions.
- [ ] **8. Interpret ρ in log space (low).** `f_H = ρ f_L + δ` in log space is
  `y_H = y_L^ρ · e^δ`, a power law. ρ far from 1 flags CNP miscalibration.
  *Fix:* print ρ from `mfgp/model.json` in run reports.
- [ ] **9. IVR wording in docs (low, docs only).** `docs/architecture.md` (Active Learning — IVR)
  calls `(1/N) Σ k²(θ_i, θ)/σ²(θ)` the integrated variance `I(θ)` to be minimized. It is the
  variance *reduction*, which is maximized. `core/optimizer.py` (`IvrAcquisition.score`) is correct.

## Checked and fine
- Logistic-normal mean approximation (3/π² vs probit π/8): only p̄ is used, so either is fine.
- `y_raw` uses target events only, disjoint from the context, so no leakage.
- Normalization and GP hyperparameters are fitted on training partitions only.

## Suggested order
1 → 2 → 4 → 3 → 5, then 6–9. Only 2 and 3 need new code; the rest are config, reporting or docs
changes.
