"""Binomial count GP: one latent hit probability fitted directly to per-voxel hit counts.

Output directory::

    observations.json        observation sets, voxel counts and data provenance
    training_arrays.npz      theta, hits, events, set index and voxel index per training voxel
    model.pkl, model.json    fitted GP and its metadata
    evaluation_<e>.npz       per evaluation set: observed fraction, predictive mean/sd/intervals
    metrics.json             one row per evaluation set
    development_map.npz      probability mean and variance at every loaded voxel
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import binom

from core.gp_metrics import gp_metrics

# Gaussian-equivalent central probabilities of the ±1/2/3 sigma bands.
LEVELS = {1: 0.682689492, 2: 0.954499736, 3: 0.997300204}


def _counts(batch):
    """Hits [B] and events [B] per voxel."""
    hits = np.where(batch.event_mask, batch.labels, 0).sum(axis=1).astype(int)
    return hits, batch.events_per_voxel.astype(int)


def run_binomial_gp(cfg, *, provider, output_directory, extra_metadata=None):
    """Fit the binomial GP on the configured count sets and score the evaluation sets."""
    from core.binomial_gp import BinomialGP
    from core.gp_subsample import subsample_voxels

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    thetas, hits, events, sets, indices, records = [], [], [], [], [], []
    for k, item in enumerate(cfg.observations):
        batch, provenance = provider.load(item.partition, item.fidelity)
        index = subsample_voxels(batch.batch_size, item.max_voxels, cfg.seed, k)
        batch = batch.take(index)
        m, n = _counts(batch)
        thetas.append(batch.theta)
        hits.append(m)
        events.append(n)
        sets.append(np.full(len(m), k))
        indices.append(index)
        records.append(dict(set=k, **item.model_dump(mode="json"), voxels=len(m),
                            hits=int(m.sum()), events=int(n.sum()), data=provenance))
    theta = np.concatenate(thetas)
    m, n = np.concatenate(hits), np.concatenate(events)
    np.savez_compressed(directory / "training_arrays.npz", theta=theta, hits=m, events=n,
                        set=np.concatenate(sets), index=np.concatenate(indices))
    (directory / "observations.json").write_text(json.dumps(records, indent=2, allow_nan=False))

    gp = BinomialGP(kernel=cfg.kernel).fit(
        theta, m, n, max_iters=cfg.max_iters, n_restarts=cfg.n_restarts, seed=cfg.seed
    )
    gp.save(directory / "model.pkl")
    model = gp.model
    metadata = dict(
        config=cfg.model_dump(mode="json"),
        likelihood="m ~ Binomial(N, sigmoid(f(theta))), Laplace posterior",
        training_voxels=len(m),
        log_likelihood=float(model.log_likelihood()),
        parameter_names=model.parameter_names_flat().tolist(),
        parameters=model.param_array.tolist(),
        optimization=[dict(status=str(r.status), function_evaluations=int(r.funct_eval))
                      for r in model.optimization_runs],
        **(extra_metadata or {}),
    )
    (directory / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))

    metrics, loaded = [], [theta]
    rng_seed = cfg.seed
    for e, item in enumerate(cfg.evaluation):
        batch, provenance = provider.load(item.partition, item.fidelity)
        k, total = _counts(batch)
        observed = k / total
        samples = gp.predict_observations(batch.theta, total, n_draws=cfg.prediction_draws,
                                          seed=rng_seed + e)               # [draws, B] fractions
        mean, sd = samples.mean(axis=0), samples.std(axis=0)
        intervals = {
            level: tuple(np.quantile(samples, [(1 - q) / 2, (1 + q) / 2], axis=0))
            for level, q in LEVELS.items()
        }
        scores = gp_metrics(observed, mean, intervals=intervals, samples=samples,
                            median=np.median(samples, axis=0))
        # Count NLL: binomial mass mixed over marginal latent draws (Monte Carlo).
        mu, variance = gp.predict_latent(batch.theta)
        rng = np.random.default_rng(np.random.SeedSequence([rng_seed, e]))
        p = expit(rng.normal(mu, np.sqrt(variance), size=(cfg.prediction_draws, len(mu))))
        log_mass = logsumexp(binom.logpmf(k, total, p), axis=0) - np.log(len(p))
        scores.update(
            evaluation=e, partition=item.partition, fidelity=item.fidelity,
            voxels=len(observed), data=provenance,
            predictive_nll=float(-log_mass.mean()),
            predictive_distribution="binomial counts mixed over logistic Gaussian latent p",
            nll_measure="count probability mass; not comparable to a density NLL",
            coverage={str(level): float(((observed >= lo) & (observed <= hi)).mean())
                      for level, (lo, hi) in intervals.items()},
        )
        metrics.append(scores)
        np.savez_compressed(
            directory / f"evaluation_{e}.npz",
            theta=batch.theta, observed=observed, hits=k, target_events=total,
            mean=mean, sigma=sd, residual=mean - observed,
            label=np.array(f"{item.partition}/{item.fidelity}"),
            **{f"{bound}_{level}": intervals[level][i]
               for level in LEVELS for i, bound in enumerate(("lower", "upper"))},
        )
        loaded.append(batch.theta)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))

    grid = np.unique(np.concatenate(loaded), axis=0)
    mean, variance = gp.predict(grid)
    np.savez_compressed(directory / "development_map.npz", theta=grid, mean=mean,
                        variance=variance)
    norm = provider.normalization()
    if norm is not None:
        (directory / "normalization.json").write_text(json.dumps(norm, indent=2))
    _plot(directory)
    return directory


def _plot(directory, coverage_ylim=None):
    """Coverage and z-score plots per evaluation set (observation-predictive).

    coverage_ylim: fixed y range of the coverage plots; None reads ``coverage_ylim`` from the
    run's model.json config, and if that is unset the range follows the data.
    """
    from viz.dispatch import plot_coverage_test
    from viz.zscore import plot_zscores

    if coverage_ylim is None and (directory / "model.json").is_file():
        saved = json.loads((directory / "model.json").read_text())
        coverage_ylim = saved.get("config", {}).get("coverage_ylim")
    for path in sorted(directory.glob("evaluation_*.npz")):
        with np.load(path) as a:
            observed, mean, sigma = a["observed"], a["mean"], a["sigma"]
            intervals = {k: (a[f"lower_{k}"], a[f"upper_{k}"]) for k in LEVELS}
            label = str(a["label"])
        high = max(float(observed.max()), float(intervals[3][1].max()))
        plot_coverage_test(
            observed, mean, sigma, out_path=directory / f"{path.stem}_coverage.png",
            title=f"{label}: binomial GP observation-predictive coverage",
            xlabel="Evaluation voxel index", predicted_label="GP mean",
            raw_label="Observed fraction",
            ylim=tuple(coverage_ylim) if coverage_ylim else (0, high * 1.05),
            intervals=intervals,
            show_predicted=False,
        )
        z = (observed - mean) / np.where(sigma > 0, sigma, np.nan)
        plot_zscores(np.nan_to_num(z), mean, directory / f"{path.stem}_zscore",
                     title=f"{label}: z = (observed − predictive mean) / predictive sd",
                     rate_label="GP predicted rate")


def run_binomial_config(config):
    """Standalone run of a BinomialGPRunConfig."""
    from core.surrogates.experiment import prepare_and_open

    provider, record = prepare_and_open(config)
    return run_binomial_gp(config, provider=provider, output_directory=config.output_directory,
                           extra_metadata=dict(data_preparation=record))
