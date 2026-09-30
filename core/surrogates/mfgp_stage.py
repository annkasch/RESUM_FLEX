"""Optional three-fidelity GP stage following event-surrogate training."""

import hashlib
import json

import numpy as np


def run_mfgp_stage(config, surrogate):
    # Keep optional GPy/Emukit dependencies out of the neural import path.
    from core.mfgp_pipeline import fit_mfgp_three_fidelity
    from core.surrogate_cnp import split_context_target
    from core.surrogate_mfgp import load_mfgp, save_mfgp
    from core.surrogates.pipeline import prepare_surrogate_datasets
    from data.optical_pipeline import load_prepared_partition, prepared_partition_paths
    from data.grouped_batches import batch_groups

    cfg = config.mfgp
    directory = config.output_directory / "mfgp"
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
    np.savez_compressed(directory / "training_arrays.npz", **data)
    state = np.random.get_state()
    try:
        np.random.seed(cfg.seed)
        gp = fit_mfgp_three_fidelity(data, kernel=cfg.kernel, n_restarts=cfg.n_restarts)
    finally:
        np.random.set_state(state)
    save_mfgp(directory / "model.pkl", gp)
    restored = load_mfgp(directory / "model.pkl")
    metadata = dict(
        config=cfg.model_dump(mode="json"),
        data=provenance,
        surrogate_checkpoint=str(config.output_directory / "checkpoints/best"),
        levels=["LF surrogate mean", "HF surrogate mean", "HF raw target fraction"],
        fit_split="train",
        evaluation_split="validation",
        test_data="Not loaded",
        coordinates="Prepared theta; use saved normalization for physical queries",
        variance="GP observation-predictive variance including fitted per-level Gaussian noise",
        noise="Learned per-fidelity constant noise; CNP scale is not used as GP noise",
        log_likelihood=float(gp.model.log_likelihood()),
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
        coordinates = np.concatenate(positions)
        observed, predicted = np.concatenate(observed_parts), np.concatenate(predicted_parts)
        level = 0 if fid == "lf" else 2
        mean, variance = gp.predict(coordinates, fidelity=level)
        loaded_mean, loaded_var = restored.predict(coordinates, fidelity=level)
        np.testing.assert_array_equal(mean, loaded_mean)
        np.testing.assert_array_equal(variance, loaded_var)
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
            observed=observed,
            cnp_mean=predicted,
            mean=mean,
            sigma=sigma,
            residual=residual,
        )
    # Highest-fidelity map at all available development coordinates, excluding test files.
    theta = np.unique(np.concatenate([g.theta for b in batches.values() for g in batch_groups(b)]), axis=0)
    mean, variance = gp.predict(theta, fidelity=2)
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
