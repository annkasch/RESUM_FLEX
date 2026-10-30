"""Multi-fidelity GP over configurable levels, after training (combined) or standalone.

Each level is one (X, Y) set, ordered from the lowest to the highest fidelity:
``model_mean`` = mean predicted p of an event model per voxel (precomputed inference file
or the model itself), ``observed_fraction`` = simulated hit fraction m/N per voxel. The
top level is the target. GP hyperparameters are fitted on the configured levels only;
evaluation partitions are scored afterwards.

Output directory::

    training_arrays.npz      X_k, Y_k, events_k, index_k per level k (original units)
    fit_arrays.npz           Y_k after log-mode smoothing/floor (X_k unchanged)
    transformed_training_arrays.npz   exact GP targets (log of fit_arrays in log mode)
    levels.json              level settings, voxel counts and data provenance
    model.pkl, model.json    fitted GP and its metadata
    evaluation_<k>.npz       per evaluation set: observed, GP mean/sigma/intervals, ...
    metrics.json             one row per evaluation set
    development_map.npz      top-level prediction at every loaded voxel
"""

import json
from pathlib import Path

import numpy as np

from core.gp_metrics import mfgp_metrics


def _model_mean(model, batch, *, n_context, seed):
    """Mean predicted p per voxel [B]; only context models split off a context."""
    from core.mfgp_pipeline import model_mean

    return model_mean(model, batch, n_context=n_context if model.uses_context else None,
                      seed=seed)


class _Data:
    """Prepared partitions of one data directory, each loaded once."""

    def __init__(self, root):
        self.root = Path(root)
        self._labeled, self._inputs = {}, {}

    def name(self, partition):
        from core.surrogates.experiment import ROLE_PARTITIONS, partition_name

        return partition_name(self.root, partition) if partition in ROLE_PARTITIONS else partition

    def labeled(self, partition, fidelity):
        """(batch, provenance) with labels."""
        from data.optical_pipeline import load_partition

        key = self.name(partition), fidelity
        if key not in self._labeled:
            self._labeled[key] = load_partition(self.root, *key)
        return self._labeled[key]

    def inputs(self, partition, fidelity):
        """(batch, provenance) for context-free predictions; labels may be a placeholder."""
        from data.optical_view import OpticalView

        key = self.name(partition), fidelity
        if key in self._labeled or not (self.root / "view.json").is_file():
            return self.labeled(partition, fidelity)
        if key not in self._inputs:
            view = OpticalView(self.root)
            self._inputs[key] = view.load_inputs(*key)[0], view.provenance(*key)
        return self._inputs[key]


def _level_arrays(level, stream, data, cfg, model):
    """X [n, D], Y [n], events [n], voxel index [n] and provenance of one level."""
    from core.gp_subsample import subsample_voxels

    seed = cfg.seed + 1 + 100 * stream  # context split of this level's voxels
    if level.source == "observed_fraction":
        batch, provenance = data.labeled(level.partition, level.fidelity)
        index = subsample_voxels(batch.batch_size, level.max_voxels, cfg.seed, stream)
        batch = batch.take(index)
        from core.mfgp_pipeline import observed_fraction

        fraction, events = observed_fraction(batch)
        return batch.theta, fraction, events, index, provenance
    if level.inference is not None:
        return _inference_arrays(level, stream, data, cfg)
    if model is None:
        raise ValueError("model_mean level without inference file needs an event model")
    path = data.root / "normalization.json"
    if path.exists():  # unnormalized synthetic batches have none
        from core.surrogates.inference import normalization_matches

        saved = getattr(model, "metadata", {}).get("normalization.json")
        if not normalization_matches(saved, json.loads(path.read_text())):
            raise ValueError(
                f"Event model was trained with another normalization than {data.root} "
                "(or does not record it)"
            )
    load = data.labeled if model.uses_context else data.inputs
    batch, provenance = load(level.partition, level.fidelity)
    index = subsample_voxels(batch.batch_size, level.max_voxels, cfg.seed, stream)
    batch = batch.take(index)
    mean = _model_mean(model, batch, n_context=cfg.n_context, seed=seed)
    events = batch.events_per_voxel.astype(float)
    if model.uses_context:
        events = events - cfg.n_context  # the mean covers the target events only
    return batch.theta, mean, events, index, provenance


def _inference_arrays(level, stream, data, cfg):
    """Level from a saved inference file, checked against the prepared partition."""
    from core.gp_subsample import subsample_voxels
    from data.optical_view import OpticalView

    path = Path(level.inference)
    meta = json.loads(path.with_suffix(".json").read_text())
    if (meta["partition"], meta["fidelity"]) != (data.name(level.partition), level.fidelity):
        raise ValueError(
            f"{path} holds {meta['partition']}/{meta['fidelity']}, not "
            f"{level.partition}/{level.fidelity}"
        )
    view = OpticalView(data.root)
    theta, paths = view.load_theta(data.name(level.partition), level.fidelity)
    with np.load(path) as arrays:
        files, saved_theta = arrays["files"].tolist(), arrays["theta"]
        mean, events = arrays["mean_probability"], arrays["event_counts"].astype(float)
    if files != paths or not np.allclose(saved_theta, theta, rtol=0, atol=1e-12):
        raise ValueError(f"{path} does not match the voxels or normalization of {data.root}")
    index = subsample_voxels(len(mean), level.max_voxels, cfg.seed, stream)
    provenance = dict(
        inference=str(path), checkpoint=meta["checkpoint"], step=meta["step"], data=meta["data"]
    )
    return theta[index], mean[index], events[index], index, provenance


def _fit_targets(level, values, events, cfg):
    """GP target in original units; log mode smooths fractions and floors model means."""
    if cfg.output_transform != "log":
        return values
    if level.source == "observed_fraction":
        a = cfg.raw_pseudocount
        return (values * events + a) / (events + 2 * a)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Log-space MFGP requires nonnegative finite model means")
    return np.maximum(values, cfg.cnp_log_floor)


def run_mfgp(
    cfg, *, data_directory, output_directory, model=None, metrics_seed=None, extra_metadata=None
):
    """Fit the configured levels, score the evaluation sets and write the output directory."""
    from core.mfgp_pipeline import fit_mfgp, observed_fraction
    from core.surrogate_mfgp import load_mfgp, save_mfgp

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = _Data(data_directory)
    metrics_seed = cfg.seed if metrics_seed is None else metrics_seed
    streams = {}  # one voxel subset per (partition, fidelity), shared by its levels
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
        record = dict(level=k, **level.model_dump(mode="json"), voxels=len(Y), data=provenance)
        if cfg.output_transform == "log" and level.source == "model_mean":
            record["log_floored"] = int((Y < cfg.cnp_log_floor).sum())  # should be 0
        records.append(record)
    np.savez_compressed(directory / "training_arrays.npz", **arrays)
    np.savez_compressed(directory / "fit_arrays.npz", **fit)
    np.savez_compressed(
        directory / "transformed_training_arrays.npz",
        **{
            key: np.log(value) if cfg.output_transform == "log" and key.startswith("Y_") else value
            for key, value in fit.items()
        },
    )
    (directory / "levels.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    n = len(cfg.levels)
    state = np.random.get_state()  # GPy restarts draw from the global numpy stream
    try:
        np.random.seed(cfg.seed)
        gp = fit_mfgp(
            [fit[f"X_{k}"] for k in range(n)],
            [fit[f"Y_{k}"] for k in range(n)],
            kernel=cfg.kernel,
            n_restarts=cfg.n_restarts,
            output_transform=cfg.output_transform,
        )
    finally:
        np.random.set_state(state)
    save_mfgp(directory / "model.pkl", gp)
    restored = load_mfgp(directory / "model.pkl")
    metadata = dict(
        config=cfg.model_dump(mode="json"),
        levels=[f"{r['source']} {r['partition']}/{r['fidelity']}" for r in records],
        target_level=n - 1,
        coordinates="Prepared theta; use saved normalization for physical queries",
        variance="Original-scale predictive variance; log mode uses lognormal moments",
        noise="Learned per-level constant noise",
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
    metrics, thetas = [], [arrays[f"X_{k}"] for k in range(n)]
    for e, evaluation in enumerate(cfg.evaluation):
        level = n - 1 if evaluation.level is None else evaluation.level
        batch, provenance = data.labeled(evaluation.partition, evaluation.fidelity)
        observed, events = observed_fraction(batch)
        mean, variance = gp.predict(batch.theta, fidelity=level)
        loaded_mean, loaded_var = restored.predict(batch.theta, fidelity=level)
        np.testing.assert_array_equal(mean, loaded_mean)
        np.testing.assert_array_equal(variance, loaded_var)
        sigma = np.sqrt(variance)
        if not np.isfinite(mean).all() or not np.isfinite(sigma).all():
            raise ValueError("Nonfinite MFGP predictions")
        intervals = {
            k: gp.predict_interval(batch.theta, fidelity=level, n_sigma=k) for k in (1, 2, 3)
        }
        scores = mfgp_metrics(gp, batch.theta, observed, fidelity=level, seed=metrics_seed)
        scores.update(
            evaluation=e,
            partition=evaluation.partition,
            fidelity=evaluation.fidelity,
            level=level,
            voxels=len(observed),
            data=provenance,
            coverage={
                str(k): float(
                    ((observed >= intervals[k][0]) & (observed <= intervals[k][1])).mean()
                )
                for k in (1, 2, 3)
            },
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
    # Target-level map at every loaded voxel (levels and evaluation sets).
    theta = np.unique(np.concatenate(thetas), axis=0)
    mean, variance = gp.predict(theta, fidelity=n - 1)
    physical = theta
    normalization = data.root / "normalization.json"
    if normalization.exists():
        norm = json.loads(normalization.read_text())
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
    return directory


def run_mfgp_stage(config, surrogate, *, checkpoint_path=None):
    """Combined mode: the GP of a training run, with the freshly trained event model."""
    checkpoint = checkpoint_path or config.output_directory / "checkpoints/best"
    return run_mfgp(
        config.mfgp,
        data_directory=config.data_directory,
        output_directory=config.output_directory / "mfgp",
        model=surrogate,
        metrics_seed=config.validation_seed,
        extra_metadata=dict(mode="combined", surrogate_checkpoint=str(checkpoint)),
    )


def run_mfgp_config(config):
    """Standalone mode: a saved checkpoint and/or inference files on prepared data."""
    if config.data_config is not None:
        from core.surrogates.experiment import ensure_matching_data

        ensure_matching_data(config.data_config, config.data_directory)
    model = None
    if config.checkpoint is not None:
        from core.surrogates.checkpoints import load_surrogate

        model = load_surrogate(config.checkpoint)
    return run_mfgp(
        config,
        data_directory=config.data_directory,
        output_directory=config.output_directory,
        model=model,
        extra_metadata=dict(
            mode="standalone",
            surrogate_checkpoint=None if config.checkpoint is None else str(config.checkpoint),
        ),
    )
