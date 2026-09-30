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
    from data.optical_pipeline import load_prepared_batch

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
        path = config.data_directory / "batches" / split / f"{fid}.npz"
        batches[split, fid] = load_prepared_batch(path)
        provenance[f"{split}/{fid}"] = dict(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    data = prepare_surrogate_datasets(
        surrogate,
        batches["train", "lf"],
        batches["train", "hf"],
        n_lf_context=cfg.n_context,
        n_hf_context=cfg.n_context,
        seed=cfg.seed,
    )
    np.savez_compressed(directory / "training_arrays.npz", **data)
    fit_data = dict(data)
    transform_details = {"output_transform": cfg.output_transform}
    if cfg.output_transform == "log":
        n_target = batches["train", "hf"].n_events - cfg.n_context
        a = cfg.raw_pseudocount
        fit_data["Y_hf_raw"] = (data["Y_hf_raw"] * n_target + a) / (n_target + 2 * a)
        for key in ("Y_lf_cnp", "Y_hf_cnp"):
            if not np.isfinite(data[key]).all() or np.any(data[key] < 0):
                raise ValueError("Log-space MFGP requires nonnegative finite CNP means")
            fit_data[key] = np.maximum(data[key], cfg.cnp_log_floor)
        transform_details.update(
            raw_smoothing="(m + a) / (N + 2a), applied only to HF training counts",
            raw_pseudocount=a, hf_target_events=n_target,
            cnp_log_floor=cfg.cnp_log_floor,
            floored_cnp_rows={key: int((data[key] < cfg.cnp_log_floor).sum())
                              for key in ("Y_lf_cnp", "Y_hf_cnp")},
            coverage_note=("Raw-fraction interval inclusion; positive lognormal intervals "
                           "cannot cover zero counts. Not binomial count coverage."),
        )
    # Persist original-scale adjusted targets and exact transformed GP targets.
    np.savez_compressed(directory / "fit_arrays.npz", **fit_data)
    np.savez_compressed(directory / "transformed_training_arrays.npz", **{
        key: np.log(value) if cfg.output_transform == "log" and key.startswith("Y_") else value
        for key, value in fit_data.items()
    })
    (directory / "transform.json").write_text(json.dumps(transform_details, indent=2))
    state = np.random.get_state()
    try:
        np.random.seed(cfg.seed)
        gp = fit_mfgp_three_fidelity(fit_data, kernel=cfg.kernel, n_restarts=cfg.n_restarts,
                                     output_transform=cfg.output_transform)
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
        output_transform=cfg.output_transform,
        variance="Original-scale predictive variance; log mode uses lognormal moments",
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
        context, target = split_context_target(
            batch, config.validation_context_events, seed=config.validation_seed
        )
        observed = target.labels.mean(1)
        predicted = surrogate.predict(target, context=context).mean
        level = 0 if fid == "lf" else 2
        mean, variance = gp.predict(batch.theta, fidelity=level)
        loaded_mean, loaded_var = restored.predict(batch.theta, fidelity=level)
        np.testing.assert_array_equal(mean, loaded_mean)
        np.testing.assert_array_equal(variance, loaded_var)
        intervals = {k: gp.predict_interval(batch.theta, fidelity=level, n_sigma=k)
                     for k in (1, 2, 3)}
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
                coverage={str(k): float(((observed >= intervals[k][0]) &
                                        (observed <= intervals[k][1])).mean())
                          for k in (1, 2, 3)},
            )
        )
        np.savez_compressed(
            directory / f"{fid}_validation.npz",
            theta=batch.theta,
            observed=observed,
            target_events=np.array(target.n_events),
            cnp_mean=predicted,
            mean=mean,
            sigma=sigma,
            residual=residual,
            output_transform=np.array(cfg.output_transform),
            **{f"{bound}_{k}": intervals[k][i]
               for k in (1, 2, 3) for i, bound in enumerate(("lower", "upper"))},
        )
    # Highest-fidelity map at all available development coordinates, excluding test files.
    theta = np.unique(np.concatenate([b.theta for b in batches.values()]), axis=0)
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
    if cfg.projections.enabled:
        from core.mfgp_projections import ensure_mfgp_projections

        ensure_mfgp_projections(directory, cfg.projections)
    return directory
