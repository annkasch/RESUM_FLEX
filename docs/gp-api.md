# GP and active learning from Python

Fitting, checking and querying the MFGP in code, and the IVR/EI loop. `cnp` below is a
trained event model ([python-api.md](python-api.md)); config-driven runs use
`python -m core.surrogates mfgp` instead ([gp.md](gp.md)).

## Fit the MFGP on your LF + HF data

```python
from core import BatchLevel, fit_mfgp, levels_from_batches, save_mfgp

lf_batch = StandardBatch(mode=InputMode.FULL, theta=lf_theta, phi=lf_phi,
                         labels=lf_X.astype(np.int8))
hf_batch = StandardBatch(mode=InputMode.FULL, theta=hf_theta, phi=hf_phi,
                         labels=hf_X.astype(np.int8))

# Levels, lowest fidelity first; here the paper's recipe LF β̄ -> HF β̄ -> HF y_raw.
# Levels on the same batch, n_context and seed share their target events.
X_list, Y_list = levels_from_batches(cnp, [
    BatchLevel("model_mean", lf_batch, n_context=lf_batch.n_events // 2, seed=1),
    BatchLevel("model_mean", hf_batch, n_context=hf_batch.n_events // 2, seed=101),
    BatchLevel("observed_fraction", hf_batch, n_context=hf_batch.n_events // 2, seed=101),
])
mfgp = fit_mfgp(X_list, Y_list, kernel=cfg.mfgp.kernel, n_restarts=5)

save_mfgp("results/mfgp.pkl", mfgp)        # pickle-backed
```

**Predict** at any θ:

```python
import numpy as np
theta_query = np.array([[0.1], [0.2], [0.3]])         # (n, D_θ)
mu, var = mfgp.predict(theta_query)                    # 1-D arrays of length n
# Or get the typed schema:
prediction = mfgp.predict_as_model_prediction(theta_query)
print(prediction.mean, prediction.variance, prediction.theta_query.shape)
```

## Held-out MFGP coverage check

```python
from core import evaluate_mfgp_coverage_from_batch

holdout_batch = StandardBatch(mode=InputMode.FULL, theta=test_theta,
                               phi=test_phi, labels=test_X.astype(np.int8))

result = evaluate_mfgp_coverage_from_batch(mfgp, cnp, holdout_batch, seed=0)

print(f"1σ coverage: {result['1sigma']:.1%}  (target 68.3%)")
print(f"2σ coverage: {result['2sigma']:.1%}  (target 95.5%)")
print(f"3σ coverage: {result['3sigma']:.1%}  (target 99.7%)")

# Plot it the same way as the CNP-only path:
from viz import plot_coverage_test
plot_coverage_test(
    y_raw=result["y_obs"],
    y_predicted=result["mu"],
    sigma_predicted=result["sigma"],
    out_path="results/plots/mfgp_coverage.png",
    title="MFGP coverage on held-out HF",
)
```

## Active learning — IVR (exploration) or EI (exploitation)

Once the MFGP is fit, run an active-learning loop to pick the next θ
to simulate. Two acquisitions are available:

```python
from core import ActiveLearningLoop, BoxBounds, HighFidelityObservation

bounds = BoxBounds(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))

def evaluate_hf(theta, *, n_events, seed):
    # Call the real simulator here, then aggregate its event output with CNP.
    beta_bar, y_raw = run_real_simulation_and_aggregate(theta, n_events, seed)
    return HighFidelityObservation(beta_bar=beta_bar, y_raw=y_raw)

loop = ActiveLearningLoop(
    mfgp=mfgp, bounds=bounds, data=data,
    observation_provider=evaluate_hf,
    n_hf_events=128, n_mc_samples=1000, n_candidates_per_axis=50,
    refit_n_restarts=5,
    acquisition="ei",          # or "ivr"
    target="min",              # for EI: 'min' or 'max'
)
records = loop.run(n_steps=5)

for rec in records:
    print(f"step {rec.step}: θ_next={rec.theta_next}  "
          f"IV {rec.integrated_variance_before:.3e} → {rec.integrated_variance_after:.3e}")
```

* `acquisition="ivr"` — pure exploration. Direction-agnostic;
  `target` is ignored at acquisition time.
* `acquisition="ei"` — Expected Improvement, exploitation-leaning.
  `target="max"` searches for an argmax, `target="min"` for an argmin.
  Stars cluster near the predicted optimum once σ shrinks.

## Output — `ModelPrediction`

What MFGP-style models return at query points:

| Field | Shape | Meaning |
|---|---|---|
| `mean` | `[B]` | Posterior mean μ(θ) |
| `variance` | `[B]` | Posterior variance σ²(θ) (≥ 0) |
| `theta_query` | `[B, D_θ]` | The θ values queried |

Available via `MultiFidelityGP.predict_as_model_prediction(X)`.

## Output — MFGP levels

Returned by `levels_from_batches` / `synthetic_levels`: `(X_list, Y_list)`, one entry per
GP level, lowest fidelity first; the last level is the target.

| Entry | Shape | Meaning |
|---|---|---|
| `X_list[k]` | `(n_k, D_θ)` | Per-trial θ of level k |
| `Y_list[k]` | `(n_k, 1)` | `model_mean` (β̄ per trial) or `observed_fraction` (`m/N`) |

## Output — coverage dict

Returned by `evaluate_mfgp_coverage_from_batch`:

| Key | Shape / Type | Meaning |
|---|---|---|
| `theta` | `(n_test, D_θ)` | Held-out θ |
| `y_obs` | `(n_test,)` | Observed `m/N` per trial |
| `mu` | `(n_test,)` | MFGP posterior mean |
| `sigma` | `(n_test,)` | MFGP posterior std |
| `1sigma` / `2sigma` / `3sigma` | float | Fraction inside ±kσ band |

Target Gaussian rates: 68.27 / 95.45 / 99.73 %.
