"""Backend-independent spatial marginalization and individual-voxel projections."""

from itertools import combinations
from math import erf, sqrt

import numpy as np
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


def project_response(adapter, training_theta, settings, *, observation_model=None):
    """Project any adapter's joint response draws on physical midpoint volume cells.

    The adapter owns normalization, link and fidelity; the observation model is
    explicitly separate. No model-specific inference belongs in this engine.
    """
    axes, edges, points, mask, bounds = projection_grid(training_theta, settings)
    chosen = mask.ravel()
    posterior = adapter.prepare(points[chosen])
    point_mean = np.asarray(posterior.mean, float)
    if point_mean.shape != (int(chosen.sum()),) or not np.isfinite(point_mean).all():
        raise ValueError("Invalid physical-response mean from projection adapter")
    full_mean = np.zeros(len(points))
    full_mean[chosen] = point_mean
    full_mean = full_mean.reshape((1, *mask.shape))
    predictive = settings.quantity == "observed_fraction"
    if predictive and observation_model is None:
        raise ValueError("Observed-fraction projections require an explicit observation model")
    if not predictive and observation_model is not None:
        raise ValueError("Latent-mean projections must not include observation noise")
    draws_by_projection = {keep: [] for keep in PROJECTIONS}
    observation_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 1]))
    rng = np.random.default_rng(settings.seed)
    for start in range(0, settings.n_draws, 256):
        size = min(256, settings.n_draws - start)
        draws = np.asarray(posterior.sample(size, rng), float)
        if draws.shape != (size, int(chosen.sum())) or not np.isfinite(draws).all():
            raise ValueError("Invalid joint response draws from projection adapter")
        if predictive:
            draws = np.asarray(observation_model.sample(draws, observation_rng), float)
            if draws.shape != (size, int(chosen.sum())) or not np.isfinite(draws).all():
                raise ValueError("Invalid draws from observation model")
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
        **posterior.metadata,
        interval_probabilities=[erf(k / sqrt(2)) for k in (1, 2, 3)],
        quantity=settings.quantity,
        observation_model=(observation_model.metadata if predictive else None),
        target_events=(observation_model.metadata.get("target_events") if predictive else None),
        uncertainty=(
            "Spatial variation + latent posterior uncertainty + "
            + observation_model.metadata.get("noise_description", "observation noise")
            + "; "
            "unconditional on nonzero-hit selection"
        )
        if predictive
        else "Latent function posterior, conditional on fitted GP hyperparameters",
        weighting=(
            "Uniform physical-volume midpoint cells within domain; normalized per projection bin"
        ),
        domain_note="Training convex hull is an empirical support region, not a tank geometry model"
        if settings.domain == "training_convex_hull"
        else "User-specified physical box",
        observation_note=(
            "Overlaid target fractions are individual voxels, not marginalized measurements"
        ),
    )
