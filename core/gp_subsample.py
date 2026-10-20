"""Seeded voxel subsets for exact GPs on large training partitions (numpy only)."""

import numpy as np


def subsample_voxels(n: int, max_voxels: int | None, seed: int, stream: int) -> np.ndarray:
    """Sorted voxel indices ``[min(n, max_voxels)]``; all voxels when ``max_voxels`` is None.

    ``stream`` separates independent draws (e.g. LF vs HF) under the same seed.
    """
    if n <= 0:
        raise ValueError("Cannot subsample an empty partition")
    if max_voxels is None or max_voxels >= n:
        return np.arange(n)
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    return np.sort(rng.choice(n, size=max_voxels, replace=False))
