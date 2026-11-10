"""Seeded design point subsets for exact GPs on large training partitions (numpy only)."""

import numpy as np


def subsample_design_points(
    n: int, max_design_points: int | None, seed: int, stream: int
) -> np.ndarray:
    """Sorted design point indices ``[min(n, max_design_points)]``.

    All design points when ``max_design_points`` is None.

    ``stream`` separates independent draws (e.g. LF vs HF) under the same seed.
    """
    if n <= 0:
        raise ValueError("Cannot subsample an empty partition")
    if max_design_points is None or max_design_points >= n:
        return np.arange(n)
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    return np.sort(rng.choice(n, size=max_design_points, replace=False))
