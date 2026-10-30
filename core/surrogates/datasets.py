"""Prepared datasets behind one small interface, so applications bring their own data.

Training, inference and the GP stage read data only through a ``DatasetProvider``. A config
names its provider by import path (``data_provider: package.module:Class``); without one,
``NpzBatches`` reads ``batches/<partition>/<fidelity>.npz`` from the data directory.

A provider is constructed as ``Class(data_directory)``. It may also define a classmethod
``prepare(data_config, data_directory) -> dict`` that (re)builds the prepared data from an
application data config when it is outdated; configs with ``data_config`` require it.
"""

import hashlib
import importlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from schemas.data_models import InputMode, StandardBatch


@runtime_checkable
class DatasetProvider(Protocol):
    root: Path

    def partitions(self) -> list[str]:
        """Partition names holding data, e.g. train/validation/test or application roles."""

    def load(self, partition: str, fidelity: str, *, labels: bool = True):
        """(StandardBatch, provenance dict) of one partition, inputs already normalized.

        ``labels=False`` must also work for inputs without outcomes; the batch labels are
        then a zero placeholder and must never be scored.
        """

    def theta(self, partition: str, fidelity: str):
        """(theta [B, D_theta] normalized, voxel ids [B]) without reading events."""

    def coordinates(self, partition: str, fidelity: str) -> np.ndarray:
        """Physical voxel coordinates [B, D] for reports and maps."""

    def normalization(self) -> dict | None:
        """Affine input normalization ({'theta': ..., 'phi': ...}) or None if unnormalized."""


def load_provider(spec: str | None, root) -> DatasetProvider:
    """Provider ``spec`` ('module:Class', None = NpzBatches) for data directory ``root``."""
    if spec is None:
        return NpzBatches(root)
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"data_provider must be 'module:Class', got {spec!r}")
    try:
        factory = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import data_provider {spec!r}: {exc}") from exc
    return factory(Path(root))


def prepare_data(spec: str | None, data_config, root) -> dict:
    """Let the provider rebuild ``root`` from ``data_config`` if outdated; returns its record."""
    provider = load_provider(spec, root)
    prepare = getattr(type(provider), "prepare", None)
    if prepare is None:
        raise ValueError(f"{type(provider).__name__} cannot prepare data from a data_config")
    return prepare(Path(data_config), Path(root))


class NpzBatches:
    """Prepared batches saved as ``batches/<partition>/<fidelity>.npz`` (synthetic, tests)."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, partition, fidelity) -> Path:
        path = self.root / "batches" / partition / f"{fidelity}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"No prepared {fidelity} data for {partition} in {self.root}")
        return path

    def partitions(self) -> list[str]:
        folder = self.root / "batches"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def load(self, partition, fidelity, *, labels=True):
        path = self._path(partition, fidelity)
        with np.load(path, allow_pickle=False) as arrays:
            batch = StandardBatch(
                mode=InputMode(str(arrays["mode"].item())),
                theta=arrays["theta"],
                labels=arrays["labels"],
                phi=arrays["phi"] if "phi" in arrays else None,
                event_counts=arrays["event_counts"] if "event_counts" in arrays else None,
            )
        provenance = dict(
            path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        return batch, provenance

    def theta(self, partition, fidelity):
        batch, _ = self.load(partition, fidelity)
        return batch.theta, [f"{partition}/{fidelity}/{i}" for i in range(batch.batch_size)]

    def coordinates(self, partition, fidelity):
        return self.theta(partition, fidelity)[0]  # stored unnormalized

    def normalization(self):
        return None
