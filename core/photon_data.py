"""Ungrouped photon data for context-free event models (the neural field).

A photon source streams raw photon records [M, 7] float32 (x, y, z, px, py, pz, label bits)
in random order; RESUM_FLEX names it by import path (``photon_source: module:Class``) and
never imports a data package directly. The LEGEND optical source is
``legend_optical_map_ml.photon_loader:PhotonStream``.
"""

import importlib
from typing import Protocol


class PhotonSource(Protocol):
    n_photons: int
    n_detected: int  # photons with label 1 (training prevalence for the log-loss skill)

    def normalization(self) -> dict: ...  # offset/scale of the 6 physical columns

    def n_batches(self, batch_size: int, pass_index: int, *, drop_last: bool = False) -> int: ...

    def batches(
        self, batch_size: int, pass_index: int = 0, *, start_batch=0, drop_last=False, device="cpu"
    ): ...

    def load(self): ...  # all photons of the split as float32 [N, 7]


def photon_labels(batch):
    """Labels [M] float of raw records [M, 7] (label bits stored in the last column)."""
    import torch

    return batch[:, 6].contiguous().view(torch.int32).to(batch.dtype)


def open_photon_source(spec: str, store, split: str, **options) -> PhotonSource:
    """Photon source ``spec`` ('module:Class') for one split of ``store``."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"photon_source must be 'module:Class', got {spec!r}")
    try:
        factory = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import photon_source {spec!r}: {exc}") from exc
    return factory(store, split, **options)
