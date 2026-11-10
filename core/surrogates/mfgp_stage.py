"""Multi-fidelity GP over configurable levels, after training (combined) or standalone.

Each level is one (X, Y) set, ordered from the lowest to the highest fidelity:
``model_mean`` = mean predicted p of an event model per design point (precomputed inference file
or the model itself), ``observed_fraction`` = simulated hit fraction m/N per design point. The
top level is the target. GP hyperparameters are fitted on the configured levels only;
evaluation partitions are scored afterwards.

Output directory::

    training_arrays.npz      X_k, Y_k, events_k, index_k per level k (original units)
    fit_arrays.npz           Y_k after log-mode smoothing/floor (X_k unchanged); log mode:
                             log_offset_k of the bias correction per observed-fraction level;
                             point_noise_k with observation_noise: binomial
    transformed_training_arrays.npz   exact GP targets (log mode: log Y_k − log_offset_k)
    levels.json              level settings, design point counts, data provenance, log_offset_mean
    model.pkl, model.json    fitted GP and its metadata
    evaluation_<k>.npz       per evaluation set: observed, GP mean/sigma/intervals,
                             quantile_z (randomized quantile residuals of the hits), ...
    metrics.json             one row per evaluation set
    development_map.npz      top-level prediction at every loaded design point
    compute.json             wall/CPU time and allocated cores per stage (core.compute)
"""

import json
from pathlib import Path

import numpy as np

from core.compute import ComputeLog
from core.gp_metrics import coverage_by_brightness, mfgp_metrics


def _model_mean(model, batch, *, n_context, seed):
    """Mean predicted p per design point [B]; only context models split off a context."""
    from core.mfgp_pipeline import model_mean

    return model_mean(model, batch, n_context=n_context if model.uses_context else None, seed=seed)


# SeedSequence key under (metrics_seed, evaluation index) for the quantile residuals.
QUANTILE_KEY = 1


def _quantile_z(gp, theta, level, observed, events, noise, cfg, bias_corrected, rng):
    """Randomized quantile residuals [B] of the hits under the GP's observation predictive.

    The GP models the observation in its fitted space as N(μ, σ²), noise included (learned,
    plus binomial point noise if configured): the fraction m/N (identity) or
    log((m + a)/(N + 2a)) − offset (log; offset = log_target_offset at the predicted rate
    when the fit was bias-corrected, 0 for rates outside (0, 1)). That Gaussian is
    discretized at half-integer counts. Returns z and the number of points without offset.
    """
    from core.mfgp_pipeline import log_target_offset
    from core.quantile_residuals import discretized_gaussian_log_tails, quantile_residuals

    hits = np.rint(observed * events)
    if not np.allclose(hits, observed * events):
        raise ValueError("Observed fractions are not counts over the event totals")
    extra_noise = {} if noise is None else dict(noise=noise)
    mu, variance = gp.predict_transformed(theta, fidelity=level, **extra_noise)
    a = cfg.raw_pseudocount
    offset, outside = np.zeros_like(mu), 0
    if cfg.output_transform == "log" and bias_corrected:
        rate, _ = gp.predict(theta, fidelity=level)
        inside = (rate > 0) & (rate < 1)
        offset[inside] = log_target_offset(rate[inside], events[inside], a)
        outside = int((~inside).sum())
    if cfg.output_transform == "log":

        def transform(count, n):
            return np.log((count + a) / (n + 2 * a)) - offset

    else:

        def transform(count, n):
            return count / n

    tails = discretized_gaussian_log_tails(hits, events, mu, np.sqrt(variance), transform)
    return quantile_residuals(*tails, rng), outside


class _Data:
    """Partitions of one dataset provider, each loaded once."""

    def __init__(self, provider):
        self.provider = provider
        self.root = provider.root
        self._cache = {}

    def labeled(self, partition, fidelity):
        """(batch, provenance) with labels."""
        key = partition, fidelity, True
        if key not in self._cache:
            self._cache[key] = self.provider.load(partition, fidelity)
        return self._cache[key]

    def inputs(self, partition, fidelity):
        """(batch, provenance) for context-free predictions; labels may be a placeholder."""
        if (partition, fidelity, True) in self._cache:
            return self._cache[partition, fidelity, True]
        key = partition, fidelity, False
        if key not in self._cache:
            self._cache[key] = self.provider.load(partition, fidelity, labels=False)
        return self._cache[key]


def _level_arrays(level, stream, data, cfg, model):
    """X [n, D], Y [n], events [n], design point index [n] and provenance of one level."""
    from core.gp_subsample import subsample_design_points

    seed = cfg.seed + 1 + 100 * stream  # context split of this level's design points
    if level.source == "observed_fraction":
        batch, provenance = data.labeled(level.partition, level.fidelity)
        index = subsample_design_points(batch.batch_size, level.max_design_points, cfg.seed, stream)
        batch = batch.take(index)
        from core.mfgp_pipeline import observed_fraction

        fraction, events = observed_fraction(batch)
        return batch.theta, fraction, events, index, provenance
    if level.inference is not None:
        return _inference_arrays(level, stream, data, cfg)
    if model is None:
        raise ValueError("model_mean level without inference file needs an event model")
    current = data.provider.normalization()
    if current is not None:  # unnormalized synthetic batches have none
        from core.surrogates.inference import normalization_matches

        saved = getattr(model, "metadata", {}).get("normalization.json")
        if not normalization_matches(saved, current):
            raise ValueError(
                f"Event model was trained with another normalization than {data.root} "
                "(or does not record it)"
            )
    load = data.labeled if model.uses_context else data.inputs
    batch, provenance = load(level.partition, level.fidelity)
    index = subsample_design_points(batch.batch_size, level.max_design_points, cfg.seed, stream)
    batch = batch.take(index)
    mean = _model_mean(model, batch, n_context=cfg.n_context, seed=seed)
    events = batch.events_per_design_point.astype(float)
    if model.uses_context:
        events = events - cfg.n_context  # the mean covers the target events only
    return batch.theta, mean, events, index, provenance


def _inference_arrays(level, stream, data, cfg):
    """Level from a saved inference file, checked against the prepared partition."""
    from core.gp_subsample import subsample_design_points

    path = Path(level.inference)
    meta = json.loads(path.with_suffix(".json").read_text())
    if (meta["partition"], meta["fidelity"]) != (level.partition, level.fidelity):
        raise ValueError(
            f"{path} holds {meta['partition']}/{meta['fidelity']}, not "
            f"{level.partition}/{level.fidelity}"
        )
    theta, ids = data.provider.theta(level.partition, level.fidelity)
    with np.load(path) as arrays:
        saved_ids, saved_theta = arrays["design_point_ids"].tolist(), arrays["theta"]
        mean, events = arrays["mean_probability"], arrays["event_counts"].astype(float)
    if saved_ids != list(ids) or not np.allclose(saved_theta, theta, rtol=0, atol=1e-12):
        raise ValueError(f"{path} does not match the design points or normalization of {data.root}")
    index = subsample_design_points(len(mean), level.max_design_points, cfg.seed, stream)
    provenance = dict(
        inference=str(path), checkpoint=meta["checkpoint"], step=meta["step"], data=meta["data"]
    )
    return theta[index], mean[index], events[index], index, provenance


def _fit_targets(level, values, events, cfg):
    """GP target in original units; log mode smooths fractions and floors model means."""
    from core.mfgp_pipeline import fit_targets

    return fit_targets(
        values,
        events,
        source=level.source,
        output_transform=cfg.output_transform,
        pseudocount=cfg.raw_pseudocount,
        floor=cfg.cnp_log_floor,
    )


def run_mfgp(
    cfg,
    *,
    provider,
    output_directory,
    model=None,
    metrics_seed=None,
    extra_metadata=None,
    compute=None,
):
    """Fit the configured levels, score the evaluation sets and write the output directory.

    ``compute`` (core.compute.ComputeLog, new by default) gets the stages data_loading, fit
    and evaluation and is written to compute.json.
    """
    from core.mfgp_pipeline import (
        count_noise,
        fit_mfgp,
        fit_mfgp_corrected,
        observed_fraction,
        resolve_max_refits,
    )
    from core.surrogate_mfgp import load_mfgp, save_mfgp

    compute = ComputeLog() if compute is None else compute
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = _Data(provider)
    metrics_seed = cfg.seed if metrics_seed is None else metrics_seed
    streams = {}  # one design point subset per (partition, fidelity), shared by its levels
    arrays, fit, records = {}, {}, []
    for k, level in enumerate(cfg.levels):
        stream = streams.setdefault((level.partition, level.fidelity), len(streams))
        X, Y, events, index, provenance = _level_arrays(level, stream, data, cfg, model)
        if X is None:
            raise ValueError("MFGP requires design coordinates theta")
        arrays.update(
            {f"X_{k}": X, f"Y_{k}": Y[:, None], f"events_{k}": events[:, None], f"index_{k}": index}
        )
        fit[f"X_{k}"], fit[f"Y_{k}"] = X, _fit_targets(level, Y, events, cfg)[:, None]
        record = dict(
            level=k, **level.model_dump(mode="json"), design_points=len(Y), data=provenance
        )
        if cfg.output_transform == "log" and level.source == "model_mean":
            record["log_floored"] = int((Y < cfg.cnp_log_floor).sum())  # should be 0
        records.append(record)
    np.savez_compressed(directory / "training_arrays.npz", **arrays)
    compute.lap("data_loading")
    n = len(cfg.levels)
    X_list, Y_list = [fit[f"X_{k}"] for k in range(n)], [fit[f"Y_{k}"] for k in range(n)]
    max_refits = resolve_max_refits(cfg.max_refits, cfg.output_transform, cfg.observation_noise)
    counted = [
        arrays[f"events_{k}"].ravel() if level.source == "observed_fraction" else None
        for k, level in enumerate(cfg.levels)
    ]
    correction, offsets, point_noise = None, [None] * n, [None] * n
    state = np.random.get_state()  # GPy restarts draw from the global numpy stream
    try:
        np.random.seed(cfg.seed)
        if max_refits and any(c is not None for c in counted):
            result = fit_mfgp_corrected(
                X_list,
                Y_list,
                events=counted,
                output_transform=cfg.output_transform,
                max_refits=max_refits,
                pseudocount=cfg.raw_pseudocount,
                point_noise=cfg.observation_noise == "binomial",
                kernel=cfg.kernel,
                n_restarts=cfg.n_restarts,
            )
            gp, offsets, point_noise = result.gp, result.offsets, result.point_noise
            correction = result.summary()
            for k, outside in enumerate(result.outside):
                if outside is not None:
                    records[k]["rates_outside_unit_interval"] = outside
        else:
            gp = fit_mfgp(
                X_list,
                Y_list,
                kernel=cfg.kernel,
                n_restarts=cfg.n_restarts,
                output_transform=cfg.output_transform,
            )
    finally:
        np.random.set_state(state)
    for k, offset in enumerate(offsets):
        if offset is not None:
            fit[f"log_offset_{k}"] = offset
            records[k]["log_offset_mean"] = float(offset.mean())
        if point_noise[k] is not None:
            fit[f"point_noise_{k}"] = point_noise[k]
            records[k]["point_noise_mean"] = float(point_noise[k].mean())
    np.savez_compressed(directory / "fit_arrays.npz", **fit)
    transformed = {}
    for key, value in fit.items():
        if cfg.output_transform == "log" and key.startswith("Y_"):
            offset = offsets[int(key[2:])]
            value = np.log(value) - (0 if offset is None else offset[:, None])
        if not key.startswith(("log_offset_", "point_noise_")):
            transformed[key] = value
    np.savez_compressed(directory / "transformed_training_arrays.npz", **transformed)
    (directory / "levels.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    save_mfgp(directory / "model.pkl", gp)
    restored = load_mfgp(directory / "model.pkl")
    metadata = dict(
        config=cfg.model_dump(mode="json"),
        levels=[f"{r['source']} {r['partition']}/{r['fidelity']}" for r in records],
        target_level=n - 1,
        coordinates="Prepared theta; use saved normalization for physical queries",
        mean="Rate; log mode: exp(mu + latent variance / 2), observation noise excluded",
        variance="Original-scale predictive variance; log mode: lognormal, noise included",
        noise=(
            "Learned per-level constant noise"
            if cfg.observation_noise == "learned"
            else "Learned per-level constant noise plus binomial count noise at the "
            "predicted rate per observed-fraction point (fit_arrays point_noise_k)"
        ),
        refits=correction,
        log_likelihood=float(gp.model.log_likelihood()),
        parameter_names=gp.model.parameter_names_flat().tolist(),
        parameters=gp.model.param_array.tolist(),
        optimization=[
            dict(status=str(r.status), function_evaluations=int(r.funct_eval))
            for r in gp.model.optimization_runs
        ],
        **(extra_metadata or {}),
    )
    (directory / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    compute.lap("fit")
    metrics, thetas = [], [arrays[f"X_{k}"] for k in range(n)]
    for e, evaluation in enumerate(cfg.evaluation):
        level = n - 1 if evaluation.level is None else evaluation.level
        batch, provenance = data.labeled(evaluation.partition, evaluation.fidelity)
        observed, events = observed_fraction(batch)
        noise = None
        if cfg.observation_noise == "binomial" and cfg.levels[level].source == "observed_fraction":
            rate, _ = gp.predict(batch.theta, fidelity=level)
            noise, outside = count_noise(
                rate,
                events,
                output_transform=cfg.output_transform,
                pseudocount=cfg.raw_pseudocount,
            )
        extra_noise = {} if noise is None else dict(noise=noise)
        mean, variance = gp.predict(batch.theta, fidelity=level, **extra_noise)
        loaded_mean, loaded_var = restored.predict(batch.theta, fidelity=level, **extra_noise)
        np.testing.assert_array_equal(mean, loaded_mean)
        np.testing.assert_array_equal(variance, loaded_var)
        sigma = np.sqrt(variance)
        if not np.isfinite(mean).all() or not np.isfinite(sigma).all():
            raise ValueError("Nonfinite MFGP predictions")
        intervals = {
            k: gp.predict_interval(batch.theta, fidelity=level, n_sigma=k, **extra_noise)
            for k in (1, 2, 3)
        }
        scores = mfgp_metrics(
            gp, batch.theta, observed, fidelity=level, seed=metrics_seed, **extra_noise
        )
        quantile_z, no_offset = _quantile_z(
            gp,
            batch.theta,
            level,
            observed,
            events,
            noise,
            cfg,
            bias_corrected=bool(correction and correction["bias_correction"]),
            rng=np.random.default_rng(np.random.SeedSequence([metrics_seed, e, QUANTILE_KEY])),
        )
        if no_offset:
            scores["quantile_z_rates_outside_unit_interval"] = no_offset
        if noise is not None:
            scores["observation_noise"] = "learned + binomial count noise at the predicted rate"
            scores["rates_outside_unit_interval"] = outside  # learned noise only there
        scores.update(
            evaluation=e,
            partition=evaluation.partition,
            fidelity=evaluation.fidelity,
            level=level,
            design_points=len(observed),
            data=provenance,
            coverage={
                str(k): float(
                    ((observed >= intervals[k][0]) & (observed <= intervals[k][1])).mean()
                )
                for k in (1, 2, 3)
            },
            coverage_by_brightness=coverage_by_brightness(observed, intervals, mean),
        )
        metrics.append(scores)
        extra = {}
        if model is not None:
            extra["model_mean"] = _model_mean(
                model, batch, n_context=cfg.n_context, seed=metrics_seed
            )
        np.savez_compressed(
            directory / f"evaluation_{e}.npz",
            theta=batch.theta,
            observed=observed,
            target_events=events,
            mean=mean,
            sigma=sigma,
            residual=mean - observed,
            quantile_z=quantile_z,
            output_transform=np.array(cfg.output_transform),
            label=np.array(f"{evaluation.partition}/{evaluation.fidelity}, level {level}"),
            **extra,
            **{
                f"{bound}_{k}": intervals[k][i]
                for k in (1, 2, 3)
                for i, bound in enumerate(("lower", "upper"))
            },
        )
        thetas.append(batch.theta)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    # Target-level map at every loaded design point (levels and evaluation sets).
    theta = np.unique(np.concatenate(thetas), axis=0)
    mean, variance = gp.predict(theta, fidelity=n - 1)
    physical = theta
    norm = data.provider.normalization()
    if norm is not None:
        (directory / "normalization.json").write_text(json.dumps(norm, indent=2))
        physical = theta * np.asarray(norm["theta"]["scale"]) + np.asarray(norm["theta"]["offset"])
    np.savez_compressed(
        directory / "development_map.npz",
        theta=theta,
        theta_physical=physical,
        mean=mean,
        variance=variance,
    )
    from viz.surrogate import plot_mfgp_run

    plot_mfgp_run(directory)
    if cfg.projections.enabled:
        from core.projection_selection import ensure_selected_projections

        ensure_selected_projections(directory, cfg.projections, backend="mfgp")
    compute.lap("evaluation")
    compute.write(directory)
    return directory


def run_mfgp_stage(config, surrogate, *, checkpoint_path=None):
    """Combined mode: the GP of a training run, with the freshly trained event model."""
    from core.surrogates.datasets import load_provider

    checkpoint = checkpoint_path or config.output_directory / "checkpoints/best"
    return run_mfgp(
        config.mfgp,
        provider=load_provider(config.data_provider, config.data_directory),
        output_directory=config.output_directory / "mfgp",
        model=surrogate,
        metrics_seed=config.validation_seed,
        extra_metadata=dict(mode="combined", surrogate_checkpoint=str(checkpoint)),
    )


def run_mfgp_config(config):
    """Standalone mode: a saved checkpoint and/or inference files on prepared data."""
    from core.surrogates.experiment import prepare_and_open

    compute = ComputeLog()
    with compute.stage("data_preparation"):
        provider, _ = prepare_and_open(config)
    model = None
    if config.checkpoint is not None:
        from core.surrogates.checkpoints import load_surrogate

        model = load_surrogate(config.checkpoint)
    return run_mfgp(
        config,
        provider=provider,
        output_directory=config.output_directory,
        model=model,
        extra_metadata=dict(
            mode="standalone",
            surrogate_checkpoint=None if config.checkpoint is None else str(config.checkpoint),
        ),
        compute=compute,
    )
