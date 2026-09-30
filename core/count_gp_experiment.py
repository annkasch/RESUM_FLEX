"""Offline optical count-GP experiment; training and prediction share one likelihood."""

import json
from math import erf, sqrt

import numpy as np

from core.gp_backends import build_count_gp_backend
from data.optical_counts import prepare_optical_counts


def run_count_gp(config, prepared=None):
    data, rows, manifest = prepared if prepared is not None else prepare_optical_counts(config.data)
    if "train" not in data or "validation" not in data:
        raise ValueError("Count-GP experiment requires training and validation data")
    out = config.output_directory
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(config.model_dump_json(indent=2))
    (out / "counts.json").write_text(json.dumps(rows, indent=2))
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    # Snapshot arrays so later raw-data/config changes cannot alter a saved result.
    for split in ("train", "validation"):
        np.savez_compressed(out / f"{split}_counts.npz", **data[split])
    train = data["train"]
    settings = config.backend
    gp = build_count_gp_backend(settings).fit(
        train["theta"],
        train["hits"],
        train["trials"],
        max_iters=settings.max_iters,
        n_restarts=settings.n_restarts,
        seed=settings.seed,
    )
    gp.save(out / "model.pkl")
    results = []
    valid = data["validation"]
    # All sources share a single p(theta); group labels are reporting only.
    for group in np.unique(valid["source_group"]):
        chosen = valid["source_group"] == group
        x, hits, n = valid["theta"][chosen], valid["hits"][chosen], valid["trials"][chosen]
        observed = hits / n
        mean, variance = gp.predict(x)
        samples = gp.predict_observations(
            x, n, n_draws=config.prediction_draws, seed=config.prediction_seed
        )
        intervals = {}
        for k in (1, 2, 3):
            tail = (1 - erf(k / sqrt(2))) / 2
            intervals[k] = np.quantile(samples, [tail, 1 - tail], axis=0, method="inverted_cdf")
        np.savez_compressed(
            out / f"{group}_validation.npz",
            theta=x,
            hits=hits,
            trials=n,
            observed=observed,
            mean=mean,
            probability_variance=variance,
            observation_sigma=samples.std(0),
            **{
                f"{bound}_{k}": intervals[k][i]
                for k in (1, 2, 3)
                for i, bound in enumerate(("lower", "upper"))
            },
        )
        results.append(
            dict(
                source_group=str(group),
                voxels=len(x),
                mean_observed=float(observed.mean()),
                mean_predicted=float(mean.mean()),
                mae=float(np.abs(mean - observed).mean()),
                rmse=float(np.sqrt(np.square(mean - observed).mean())),
                coverage={
                    str(k): dict(
                        inside=int(((observed >= lo) & (observed <= hi)).sum()), total=len(x)
                    )
                    for k, (lo, hi) in intervals.items()
                },
            )
        )
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    metadata = dict(
        backend=gp.backend,
        link="logistic",
        likelihood="Binomial(m | N, p(theta))",
        inference="Laplace approximation; plug-in optimized kernel hyperparameters",
        coordinates="Physical voxel centers in meters; training-only scaling in checkpoint",
        observation_noise="Binomial only; no Gaussian observation-noise parameter",
        source_groups="LF/HF identify budgets/splits, not different latent functions",
        zero_policy="Use selected folders as configured; no label-dependent filtering here",
        log_evidence=float(gp.model.log_likelihood()),
        parameter_names=gp.model.parameter_names_flat().tolist(),
        parameters=gp.model.param_array.tolist(),
        optimization=[
            dict(status=str(r.status), function_evaluations=int(r.funct_eval))
            for r in gp.model.optimization_runs
        ],
    )
    (out / "model.json").write_text(json.dumps(metadata, indent=2))
    from viz.count_gp import plot_count_gp

    plot_count_gp(out)
    return out
