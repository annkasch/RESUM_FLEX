"""Bias of the MFGP rate prediction on synthetic counts with a known rate.

A smooth 2-D rate surface ``p(θ)`` that varies ~20× between dim and bright design points.
The low level is a model mean ``p(θ) · exp(small smooth distortion)`` at many points
(no count noise); the top level is the observed fraction ``m/N`` with
``m ~ Binomial(N, p)`` at fewer points, ``N`` chosen for a given mean number of hits.
The GP is fitted with the same target preparation as the MFGP stage
(:func:`core.mfgp_pipeline.fit_targets`) and scored on held-out points against the true
rate (bias) and against fresh observed fractions (coverage, also for dim and bright
points separately). In log mode the top-level
target gets the stage's bias correction (fit_mfgp_log_corrected) unless
``log_bias_max_refits=0``.

:func:`log_target_terms` gives the exact expectations behind the log-mode bias, per point:

* ``pseudocount_shift`` = E[y]/p − 1 with y = (m + a)/(N + 2a),
* ``jensen_shift``      = exp(E[log y])/E[y] − 1 (what the GP mean in log space sees),
* ``log_variance``      = Var[log y] (the count noise in log units).

``scripts/mfgp_noise_sweep.py`` runs :func:`noise_study` over hits per point and both
output transforms.
"""

from __future__ import annotations

import numpy as np

from core.mfgp_pipeline import (
    binomial_log_moments,
    fit_mfgp,
    fit_mfgp_log_corrected,
    fit_targets,
    log_count_noise,
    resolve_log_bias_refits,
)


def log_target_terms(probability, events, pseudocount):
    """Exact binomial expectations of the smoothed log target; arrays [n] in, dict of [n].

    probability, events: [n] true rate and event count per point, 0 < p < 1, N ≥ 1.
    """
    p = np.asarray(probability, dtype=float)
    n = np.asarray(events, dtype=float)
    mean_log, var_log = binomial_log_moments(p, n, pseudocount)
    mean_y = (n * p + pseudocount) / (n + 2 * pseudocount)
    return dict(
        pseudocount_shift=mean_y / p - 1,
        jensen_shift=np.exp(mean_log - np.log(mean_y)) - 1,
        log_variance=var_log,
    )


def rate_surface(theta, base_rate=0.01):
    """True rate p(θ) [n] for θ [n, 2] in [0, 1]²; dim/bright ratio e³ ≈ 20."""
    theta = np.asarray(theta, dtype=float)
    return base_rate * np.exp(1.5 * np.sin(2 * np.pi * theta[:, 0]) * np.cos(np.pi * theta[:, 1]))


def _model_mean(theta, base_rate):
    """Low level [n]: the true rate with a smooth multiplicative distortion (model error)."""
    return rate_surface(theta, base_rate) * np.exp(0.15 * np.cos(3 * theta[:, 0] + theta[:, 1]))


def noise_study(
    hits: float,
    *,
    output_transform: str,
    n_low: int = 300,
    n_top: int = 80,
    n_test: int = 200,
    base_rate: float = 0.01,
    pseudocount: float = 0.5,
    kernel: str = "rbf",
    n_restarts: int = 2,
    log_bias_max_refits: int | None = None,
    observation_noise: str = "learned",
    seed: int = 0,
) -> dict:
    """Fit one two-level MFGP with ``hits`` mean hits per top-level point; return scores.

    Relative biases are ratios of means minus one over the held-out points (all, dim =
    below the median true rate, bright = above), as in metrics.json mean_bias/mean_observed.
    ``log_bias_max_refits``: largest number of refits of the log-target bias correction as
    in the MFGP stage (None: 8 in log mode, 0 in identity mode; identity allows only 0).
    ``observation_noise``: learned or binomial (log mode with refits), as in the stage.
    """
    if observation_noise not in ("learned", "binomial"):
        raise ValueError("observation_noise must be learned or binomial")
    max_refits = resolve_log_bias_refits(log_bias_max_refits, output_transform)
    if hits <= 0:
        raise ValueError("hits must be positive")

    def rng(k):
        return np.random.default_rng(np.random.SeedSequence([seed, k]))

    theta_low = rng(0).uniform(size=(n_low, 2))
    theta_top = rng(1).uniform(size=(n_top, 2))
    theta_test = rng(2).uniform(size=(n_test, 2))
    p_top, p_test = rate_surface(theta_top, base_rate), rate_surface(theta_test, base_rate)
    events = max(1, round(hits / p_top.mean()))
    n_top_events = np.full(n_top, float(events))
    fraction = rng(3).binomial(events, p_top) / events
    low = _model_mean(theta_low, base_rate)
    targets = [
        fit_targets(low, np.ones(n_low), source="model_mean", output_transform=output_transform),
        fit_targets(
            fraction,
            n_top_events,
            source="observed_fraction",
            output_transform=output_transform,
            pseudocount=pseudocount,
        ),
    ]
    state = np.random.get_state()  # GPy restarts draw from the global numpy stream
    try:
        np.random.seed(seed)
        X_list, Y_list = [theta_low, theta_top], [t[:, None] for t in targets]
        correction = None
        if max_refits:
            result = fit_mfgp_log_corrected(
                X_list,
                Y_list,
                events=[None, n_top_events],
                max_refits=max_refits,
                point_noise=observation_noise == "binomial",
                pseudocount=pseudocount,
                kernel=kernel,
                n_restarts=n_restarts,
            )
            gp, correction = result.gp, result.summary()
        elif observation_noise == "binomial":
            raise ValueError("observation_noise binomial needs log mode with refits")
        else:
            gp = fit_mfgp(
                X_list,
                Y_list,
                kernel=kernel,
                n_restarts=n_restarts,
                output_transform=output_transform,
            )
    finally:
        np.random.set_state(state)
    extra = {}
    if observation_noise == "binomial":
        rate, _ = gp.predict(theta_test, fidelity=1)
        extra["noise"] = log_count_noise(rate, np.full(n_test, float(events)), pseudocount)
    mean, _ = gp.predict(theta_test, fidelity=1, **extra)
    observed = rng(4).binomial(events, p_test) / events
    dim = p_test < np.median(p_test)

    def bias(mask):
        return float(mean[mask].mean() / p_test[mask].mean() - 1)

    noise = [float(lik.variance[0]) for lik in gp.model.likelihood.likelihoods_list]
    row = dict(
        output_transform=output_transform,
        hits=float(hits),
        events=events,
        mean_hits_dim=float(events * p_test[dim].mean()),
        seed=seed,
        observation_noise=observation_noise,
        log_bias_correction=correction,
        rate_bias=bias(np.ones(n_test, dtype=bool)),
        rate_bias_dim=bias(dim),
        rate_bias_bright=bias(~dim),
        coverage={},
        coverage_dim={},
        coverage_bright={},
        top_noise_variance=noise[-1],
    )
    for k in (1, 2, 3):
        lo, hi = gp.predict_interval(theta_test, fidelity=1, n_sigma=k, **extra)
        inside = (observed >= lo) & (observed <= hi)
        row["coverage"][str(k)] = float(inside.mean())
        row["coverage_dim"][str(k)] = float(inside[dim].mean())
        row["coverage_bright"][str(k)] = float(inside[~dim].mean())
    if output_transform == "log":
        terms = log_target_terms(p_top, n_top_events, pseudocount)
        row.update(
            pseudocount_shift=float(terms["pseudocount_shift"].mean()),
            jensen_shift=float(terms["jensen_shift"].mean()),
            count_log_variance=float(terms["log_variance"].mean()),
            back_conversion_shift=float(np.expm1(noise[-1] / 2)),
        )
    return row
