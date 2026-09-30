"""Optional pooled or event-count fidelity GP stage following event-surrogate training."""

import hashlib
import json

import numpy as np


def run_mfgp_stage(config, surrogate, *, output_directory=None):
    # Keep optional GPy/Emukit dependencies out of the neural import path.
    from core.surrogate_cnp import split_context_target
    from core.surrogate_mfgp import MultiFidelityGP, load_mfgp, save_mfgp
    from core.surrogates.pipeline import prepare_surrogate_datasets, mfgp_level_arrays
    from data.optical_pipeline import load_prepared_partition, prepared_partition_paths
    from data.grouped_batches import batch_groups

    cfg = config.mfgp
    from pathlib import Path

    directory = config.output_directory / "mfgp" if output_directory is None else Path(output_directory)
    directory.mkdir(exist_ok=False)
    batches, provenance = {}, {}
    partitions = [
        ("train", "lf"),
        ("train", "hf"),
        ("validation", "lf"),
        ("validation", "hf"),
    ]
    for split, fid in partitions:
        if (split, fid) == ("validation", "lf") and not config.lf_validation:
            continue
        batches[split, fid] = load_prepared_partition(config.data_directory, split, fid)
        paths = prepared_partition_paths(config.data_directory, split, fid)
        records = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                   for path in paths]
        provenance[f"{split}/{fid}"] = records[0] if len(records) == 1 else records
    data = prepare_surrogate_datasets(
        surrogate,
        batches["train", "lf"],
        batches["train", "hf"],
        n_lf_context=cfg.n_context,
        n_hf_context=cfg.n_context,
        seed=cfg.seed,
    )
    event_counts = np.concatenate([
        np.full(g.batch_size, g.n_events, dtype=int) for g in batch_groups(batches["train", "lf"])])
    xs, ys, level_names, lf_mapping = mfgp_level_arrays(data, event_counts, lf_levels=cfg.lf_levels)
    for group in batch_groups(batches.get(("validation", "lf"), [])):
        if cfg.lf_levels == "pooled":
            lf_mapping[group.n_events] = 0
        elif group.n_events not in lf_mapping:
            raise ValueError("LF validation event count has no matching GP training level")
    np.savez_compressed(directory / "training_arrays.npz", **data,
                        lf_event_counts=event_counts,
                        **{f"X_level_{i}": x for i, x in enumerate(xs)},
                        **{f"Y_level_{i}": y for i, y in enumerate(ys)})
    state = np.random.get_state()
    try:
        np.random.seed(cfg.seed)
        gp = MultiFidelityGP(n_fidelities=len(xs), dim_theta=xs[0].shape[1],
                             kernel=cfg.kernel).fit(xs, ys, n_restarts=cfg.n_restarts)
    finally:
        np.random.set_state(state)
    save_mfgp(directory / "model.pkl", gp)
    restored = load_mfgp(directory / "model.pkl")
    metadata = dict(
        config=cfg.model_dump(mode="json"),
        data=provenance,
        surrogate_checkpoint=str(config.output_directory / "checkpoints/best"),
        levels=level_names,
        level_observations=[len(x) for x in xs],
        lf_event_count_to_level=lf_mapping,
        fit_split="train",
        evaluation_split="validation",
        test_data="Not loaded",
        coordinates="Prepared theta; use saved normalization for physical queries",
        variance="GP observation-predictive variance including fitted per-level Gaussian noise",
        noise="Learned per-fidelity constant noise; CNP scale is not used as GP noise",
        log_likelihood=float(gp.model.log_likelihood()),
        optimization_runs=[dict(status=str(r.status),
                                function_evaluations=int(r.funct_eval),
                                objective=float(r.f_opt))
                           for r in gp.model.optimization_runs],
        parameter_names=gp.model.parameter_names(),
        parameters=gp.model.param_array.tolist(),
    )
    (directory / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    metrics = []
    for fid in ("lf", "hf"):
        if ("validation", fid) not in batches:
            continue
        batch = batches["validation", fid]
        positions, observed_parts, predicted_parts = [], [], []
        for group in batch_groups(batch):
            context, target = split_context_target(
                group, config.validation_context_events, seed=config.validation_seed)
            positions.append(group.theta)
            observed_parts.append(target.labels.mean(1))
            predicted_parts.append(surrogate.predict(target, context=context).mean)
        levels = [lf_mapping[g.n_events] if fid == "lf" else gp.n_fidelities - 1
                  for g in batch_groups(batch)]
        coordinates = np.concatenate(positions)
        observed, predicted = np.concatenate(observed_parts), np.concatenate(predicted_parts)
        predictions = [gp.predict(x, fidelity=k) for x, k in zip(positions, levels, strict=True)]
        mean = np.concatenate([p[0] for p in predictions])
        variance = np.concatenate([p[1] for p in predictions])
        loaded = [restored.predict(x, fidelity=k) for x, k in zip(positions, levels, strict=True)]
        np.testing.assert_array_equal(mean, np.concatenate([p[0] for p in loaded]))
        np.testing.assert_array_equal(variance, np.concatenate([p[1] for p in loaded]))
        level = levels[0] if len(set(levels)) == 1 else sorted(set(levels))
        sigma = np.sqrt(variance)
        residual = mean - observed
        if not np.isfinite(mean).all() or not np.isfinite(sigma).all():
            raise ValueError("Nonfinite MFGP predictions")
        metrics.append(
            dict(
                fidelity=fid,
                level=level,
                voxels=len(mean),
                mean_observed=float(observed.mean()),
                mean_predicted=float(mean.mean()),
                mae=float(np.abs(residual).mean()),
                rmse=float(np.sqrt(np.square(residual).mean())),
                pearson_r=float(np.corrcoef(mean, observed)[0, 1])
                if mean.std() and observed.std()
                else None,
                coverage={str(k): float((np.abs(residual) <= k * sigma).mean()) for k in (1, 2, 3)},
            )
        )
        np.savez_compressed(
            directory / f"{fid}_validation.npz",
            theta=coordinates,
            fidelity_levels=np.concatenate([np.full(len(x), k, dtype=int)
                                             for x, k in zip(positions, levels, strict=True)]),
            observed=observed,
            cnp_mean=predicted,
            mean=mean,
            sigma=sigma,
            residual=residual,
        )
    # Highest-fidelity map at all available development coordinates, excluding test files.
    theta = np.unique(np.concatenate([g.theta for b in batches.values() for g in batch_groups(b)]), axis=0)
    mean, variance = gp.predict(theta, fidelity=gp.n_fidelities - 1)
    physical = theta
    normalization = config.data_directory / "normalization.json"
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
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    from viz.surrogate import plot_mfgp_run

    plot_mfgp_run(directory)
    return directory
