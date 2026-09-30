"""Correlated posterior-draw marginalization of a three-coordinate GP.

Supports RESOLVE-style uncertainty in spatial averages and predictive fractions
for individual voxels. Both use joint latent GP draws and midpoint volume weights.
Predictive fractions add binomial counting noise, never GP likelihood noise.
"""

import hashlib
import json
from itertools import combinations
from math import erf, sqrt
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky
from scipy.spatial import ConvexHull, QhullError

PROJECTIONS = [(0,), (1,), (2,), *combinations(range(3), 2)]


def projection_grid(training_theta, settings):
    """Uniform midpoint cells, optionally masked to the training convex hull."""
    training_theta = np.asarray(training_theta, dtype=float)
    if training_theta.ndim != 2 or training_theta.shape[1] != 3:
        raise ValueError("Spatial projections require three physical coordinates")
    if not np.isfinite(training_theta).all():
        raise ValueError("Training coordinates must be finite")
    bounds = np.asarray(
        settings.bounds
        if settings.bounds is not None
        else np.column_stack((training_theta.min(0), training_theta.max(0)))
    )
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Projection volume must have nonzero extent in all three coordinates")
    n = settings.grid_steps
    edges = [np.linspace(lo, hi, n + 1) for lo, hi in bounds]
    axes = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    mask = np.ones(len(points), dtype=bool)
    if settings.domain == "training_convex_hull":
        try:
            hull = ConvexHull(training_theta)
        except QhullError as exc:
            raise ValueError("Training centers do not span a 3D convex hull") from exc
        mask = np.all(points @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-10, axis=1)
    if not mask.any():
        raise ValueError("Projection grid contains no points in the selected domain")
    return axes, edges, points, mask.reshape((n, n, n)), bounds


def average_grid(values, mask, keep):
    """Average uniform-volume valid cells, keeping leading draw and chosen axes."""
    axes = tuple(i + 1 for i in range(3) if i not in keep)
    counts = mask.sum(axis=tuple(i for i in range(3) if i not in keep))
    total = np.where(mask[None], values, 0).sum(axis=axes)
    result = np.full(total.shape, np.nan)
    np.divide(total, counts, out=result, where=counts > 0)
    return result


def sample_locations(values, mask, keep, rng):
    """One uniformly chosen valid location per retained bin and posterior draw."""
    shape = tuple(mask.shape[i] for i in keep)
    result = np.full((len(values), *shape), np.nan)
    for index in np.ndindex(shape):
        selector = [slice(None)] * 3
        for axis, coordinate in zip(keep, index, strict=True):
            selector[axis] = coordinate
        valid = mask[tuple(selector)].ravel()
        if valid.any():
            candidates = values[(slice(None), *selector)].reshape(len(values), -1)[:, valid]
            selected = rng.integers(candidates.shape[1], size=len(values))
            result[(slice(None), *index)] = candidates[np.arange(len(values)), selected]
    return result


def marginalize_posterior(gp, training_theta, offset, scale, settings):
    """Return original-unit mean/quantiles of conditional uniform-volume averages."""
    axes, edges, points, mask, bounds = projection_grid(training_theta, settings)
    chosen = mask.ravel()
    mean, covariance = gp.predict_joint_transformed((points[chosen] - offset) / scale)
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Nonfinite GP posterior in projection volume")
    diagonal = np.diag(covariance)
    jitter = 0.0
    for attempt in range(8):
        try:
            factor = cholesky(covariance + jitter * np.eye(len(mean)), lower=True)
            break
        except np.linalg.LinAlgError:
            jitter = max(float(np.max(np.abs(diagonal))), 1.0) * 10.0 ** (-12 + attempt)
    else:
        raise ValueError("Joint posterior covariance is not positive semidefinite")
    log_mode = getattr(gp, "output_transform", "identity") == "log"
    with np.errstate(over="raise", invalid="raise"):
        point_mean = np.exp(mean + np.maximum(diagonal, 0) / 2) if log_mode else mean
    full_mean = np.zeros(len(points))
    full_mean[chosen] = point_mean
    full_mean = full_mean.reshape((1, *mask.shape))
    predictive = settings.quantity == "observed_fraction"
    if predictive and settings.target_events is None:
        raise ValueError("Observed-fraction projections require target_events")
    draws_by_projection = {keep: [] for keep in PROJECTIONS}
    observation_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 1]))
    rng = np.random.default_rng(settings.seed)
    for start in range(0, settings.n_draws, 256):
        size = min(256, settings.n_draws - start)
        draws = mean + rng.standard_normal((size, len(mean))) @ factor.T
        if log_mode:
            with np.errstate(over="raise", invalid="raise"):
                draws = np.exp(draws)
        if predictive:
            if np.any((draws < 0) | (draws > 1)):
                raise ValueError(
                    "GP latent draws fall outside [0,1]; cannot use as binomial probabilities. "
                    "Use a probability-bounded model; predictions are not silently clipped."
                )
            draws = observation_rng.binomial(settings.target_events, draws) / settings.target_events
        full = np.zeros((size, len(points)))
        full[:, chosen] = draws
        full = full.reshape((size, *mask.shape))
        for keep in PROJECTIONS:
            draws_by_projection[keep].append(
                sample_locations(full, mask, keep, observation_rng)
                if predictive
                else average_grid(full, mask, keep)
            )
    arrays = {f"axis_{i}": axis for i, axis in enumerate(axes)}
    arrays.update({f"edges_{i}": edge for i, edge in enumerate(edges)})
    arrays.update(domain_mask=mask, bounds=bounds)
    for keep, chunks in draws_by_projection.items():
        key = "".join(map(str, keep))
        samples = np.concatenate(chunks)
        valid = np.isfinite(samples[0])
        arrays[f"mean_{key}"] = average_grid(full_mean, mask, keep)[0]
        for k in (1, 2, 3):
            tail = (1 - erf(k / sqrt(2))) / 2
            low, high = np.full(valid.shape, np.nan), np.full(valid.shape, np.nan)
            low[valid], high[valid] = np.quantile(
                samples[:, valid],
                [tail, 1 - tail],
                axis=0,
                method="inverted_cdf" if predictive else "linear",
            )
            arrays[f"lower_{k}_{key}"] = low
            arrays[f"upper_{k}_{key}"] = high
    return arrays, dict(
        bounds=bounds.tolist(),
        valid_grid_points=int(chosen.sum()),
        total_grid_points=len(points),
        cholesky_jitter=jitter,
        interval_probabilities=[erf(k / sqrt(2)) for k in (1, 2, 3)],
        quantity=settings.quantity,
        target_events=settings.target_events if predictive else None,
        uncertainty=(
            "Spatial variation + latent posterior uncertainty + binomial counting noise; "
            "unconditional on nonzero-hit selection"
        )
        if predictive
        else "Latent function posterior, conditional on fitted GP hyperparameters",
        weighting=(
            "Uniform physical-volume midpoint cells within domain; normalized per projection bin"
        ),
        transform=(
            "Exponentiate latent draws, sample locations and binomial counts"
            if predictive and log_mode
            else "Exponentiate each joint draw before averaging"
            if log_mode
            else "Identity"
        ),
        domain_note="Training convex hull is an empirical support region, not a tank geometry model"
        if settings.domain == "training_convex_hull"
        else "User-specified physical box",
        observation_note=(
            "Overlaid raw HF target fractions are individual voxels, not marginalized measurements"
        ),
    )


def ensure_mfgp_projections(directory, settings):
    """Create cached projections from a saved run; never retrain or reread raw data."""
    from core.surrogate_mfgp import load_mfgp
    from viz.mfgp_projections import plot_mfgp_projections

    directory = Path(directory)
    sources = [
        directory / name for name in ("model.pkl", "training_arrays.npz", "hf_validation.npz")
    ]
    if settings.quantity == "observed_fraction" and settings.target_events is None:
        with np.load(directory / "hf_validation.npz") as validation:
            n_target = int(validation["target_events"]) if "target_events" in validation else None
        if n_target is None:
            # Older runs: recover N only from the exact prepared batch used in that run.
            model_meta = json.loads((directory / "model.json").read_text())
            experiment = json.loads((directory.parent / "experiment.json").read_text())
            record = model_meta["data"]["validation/hf"]
            batch_path = Path(record["path"])
            if (
                not batch_path.exists()
                or hashlib.sha256(batch_path.read_bytes()).hexdigest() != record["sha256"]
            ):
                raise ValueError(
                    "Cannot recover saved target N; set projections.target_events explicitly"
                )
            with np.load(batch_path) as batch:
                n_target = (
                    batch["labels"].shape[1]
                    - experiment["resolved_config"]["validation_context_events"]
                )
        if n_target <= 0:
            raise ValueError("Saved target-event count must be positive")
        settings = settings.model_copy(update={"target_events": n_target})
    norm_path = directory / "normalization.json"
    if norm_path.exists():
        sources.append(norm_path)
    signature = dict(
        version=3,
        config=settings.model_dump(mode="json"),
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    output = directory / (
        "projections_observed_fraction"
        if settings.quantity == "observed_fraction"
        else "projections"
    )
    metadata_path = output / "metadata.json"
    data_path = output / "projections.npz"
    valid = False
    if metadata_path.exists() and data_path.exists():
        valid = json.loads(metadata_path.read_text()).get("signature") == signature
    if not valid:
        offset, scale = np.zeros(3), np.ones(3)
        if norm_path.exists():
            norm = json.loads(norm_path.read_text())["theta"]
            offset, scale = np.asarray(norm["offset"]), np.asarray(norm["scale"])
        if offset.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError("Projections require a valid three-coordinate affine normalization")
        with np.load(directory / "training_arrays.npz") as data:
            training_theta = np.concatenate((data["X_lf"], data["X_hf"])) * scale + offset
            observed_train_theta = data["X_hf"] * scale + offset
            observed_train = data["Y_hf_raw"].ravel()
        gp = load_mfgp(directory / "model.pkl")
        arrays, metadata = marginalize_posterior(gp, training_theta, offset, scale, settings)
        arrays.update(observed_train_theta=observed_train_theta, observed_train=observed_train)
        with np.load(directory / "hf_validation.npz") as data:
            arrays.update(
                observed_validation_theta=data["theta"] * scale + offset,
                observed_validation=data["observed"],
            )
        output.mkdir(exist_ok=True)
        np.savez_compressed(data_path, **arrays)
        metadata_path.write_text(json.dumps(dict(signature=signature, **metadata), indent=2))
    names = ("curves", "curves_observed", "planes", "planes_observed")
    missing_counts = settings.quantity == "observed_fraction" and (
        not (output / "coverage_counts.json").exists()
        or json.loads((output / "coverage_counts.json").read_text()).get("plot_version") != 2
    )
    if (
        not valid
        or missing_counts
        or any(not (output / f"{name}.{ext}").exists() for name in names for ext in ("png", "pdf"))
    ):
        plot_mfgp_projections(output)
    return output
