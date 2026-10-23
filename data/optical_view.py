"""Per-config view on the shared event cache and the loader that builds batches from it.

A view is small: which cached files belong to which partition and fidelity, plus the
settings and normalization needed to turn their events into model inputs::

    view.npy    one row per file (VIEW_DTYPE), sorted by partition, fidelity and file name
    view.json   version, cache directory, folders, partition names, data config, normalization

Features are computed when a partition is loaded, so changing the feature settings only
needs a new view, never a new copy of the events.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from data.event_cache import NAME_BYTES, EventCache
from data.optical_features import AffineTransform, phi_features, theta_features
from schemas.data_models import InputMode, StandardBatch
from schemas.optical import OpticalDataConfig

VERSION = 1
FIDELITIES = ("lf", "hf")
VIEW_DTYPE = np.dtype(
    [
        ("partition", "i1"),  # position in view.json "partitions"
        ("fidelity", "i1"),  # position in FIDELITIES
        ("folder", "i4"),  # position in view.json "folders"
        ("name", f"S{NAME_BYTES}"),
        ("sha256", "u1", (32,)),
    ]
)
CHUNK_EVENTS = 4_000_000  # events whose features are computed at once (~200 MB float64)


def write_view(directory: Path, entries, partitions, config, normalization):
    """Write view.npy and view.json for ``entries``: (partition, CachedRun) pairs."""
    cache = entries[0][1].cache
    order = {name: i for i, name in enumerate(partitions)}
    entries = sorted(
        entries, key=lambda e: (order[e[0]], FIDELITIES.index(e[1].fidelity), e[1].file)
    )
    folders = {}
    view = np.zeros(len(entries), dtype=VIEW_DTYPE)
    for i, (partition, run) in enumerate(entries):
        folder, name = os.path.split(run.path)
        view[i] = (
            order[partition],
            FIDELITIES.index(run.fidelity),
            folders.setdefault(folder, len(folders)),
            name.encode(),
            cache.index["sha256"][run.row],
        )
    meta = {
        "version": VERSION,
        "cache": str(cache.directory),
        "folders": list(folders),
        "partitions": list(partitions),
        "fidelities": list(FIDELITIES),
        "config": config.model_dump(mode="json"),
        "normalization": normalization,
    }
    directory.mkdir(parents=True, exist_ok=True)
    # view.json is written last and marks a complete view.
    (directory / "view.json").unlink(missing_ok=True)
    np.save(directory / "view.npy", view)
    temporary = directory / "view.json.tmp"
    temporary.write_text(json.dumps(meta, indent=1) + "\n")
    temporary.replace(directory / "view.json")


class OpticalView:
    """Read access to a saved view."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "view.json").read_text())
        if self.meta.get("version") != VERSION:
            raise ValueError(f"Unsupported optical view version in {self.directory}")
        self.rows = np.load(self.directory / "view.npy")
        self.config = OpticalDataConfig.model_validate(self.meta["config"])
        self._cache = None

    @property
    def cache(self) -> EventCache:
        if self._cache is None:
            self._cache = EventCache(self.meta["cache"])
        return self._cache

    @property
    def partitions(self) -> list[str]:
        return self.meta["partitions"]

    def available(self) -> set[tuple[str, str]]:
        """(partition, fidelity) pairs holding at least one file."""
        pairs = zip(self.rows["partition"].tolist(), self.rows["fidelity"].tolist(), strict=True)
        return {(self.partitions[p], FIDELITIES[f]) for p, f in set(pairs)}

    def select(self, partition: str, fidelity: str) -> np.ndarray:
        if partition not in self.partitions:
            return self.rows[:0]
        mask = (self.rows["partition"] == self.partitions.index(partition)) & (
            self.rows["fidelity"] == FIDELITIES.index(fidelity)
        )
        return self.rows[mask]

    def paths(self, rows) -> list[str]:
        folders = self.meta["folders"]
        return [os.path.join(folders[f], n.decode()) for f, n in rows[["folder", "name"]]]

    def provenance(self, partition: str, fidelity: str) -> dict:
        """Identity of a partition's inputs: file contents, settings and normalization."""
        digest = hashlib.sha256(self.select(partition, fidelity).tobytes())
        digest.update((self.directory / "view.json").read_bytes())
        return dict(
            path=str((self.directory / "view.npy").resolve()),
            partition=partition,
            sha256=digest.hexdigest(),
        )

    def load(self, partition: str, fidelity: str) -> StandardBatch:
        """Normalized batch of one partition; ragged event counts are zero-padded."""
        view = self.select(partition, fidelity)
        if not len(view):
            raise FileNotFoundError(f"{self.directory}: no {fidelity} files in {partition}")
        cache = self.cache
        rows = cache.locate(self.paths(view), view["sha256"])
        return build_batch(cache, rows, self.config, self.meta["normalization"])


def voxel_centers(cache: EventCache, rows, config: OpticalDataConfig) -> np.ndarray:
    """Voxel centers [B, 3] used for theta and vertex offsets."""
    field = "nominal_center" if config.theta.center_source == "filename" else "measured_center"
    return cache.index[field][rows]


def phi_chunks(cache: EventCache, rows, centers, config: OpticalDataConfig):
    """Unnormalized event features of files ``rows`` in chunks of whole files.

    Yields (start, stop, values): files ``rows[start:stop]`` and their concatenated
    features [E, D_phi] in float64. centers: [B, 3] voxel centers of the files.
    """
    counts = cache.index["n_events"][rows].astype(np.int64)
    start = 0
    while start < len(rows):
        stop = start + max(1, int(np.searchsorted(np.cumsum(counts[start:]), CHUNK_EVENTS)))
        events = np.concatenate([cache.events(r) for r in rows[start:stop]])  # [E, 6]
        repeated = np.repeat(centers[start:stop], counts[start:stop], axis=0)  # [E, 3]
        try:
            values = phi_features(events[:, :3], events[:, 3:], repeated, config)[0]
        except ValueError as exc:
            raise ValueError(f"{cache.path(rows[start])} or a following file: {exc}") from exc
        yield start, stop, values
        start = stop


def build_batch(cache: EventCache, rows, config: OpticalDataConfig, normalization: dict):
    """StandardBatch of cached files ``rows``; phi is float32, as the models compute."""
    counts = cache.index["n_events"][rows].astype(np.int64)  # [B]
    width = int(counts.max())
    centers = voxel_centers(cache, rows, config)  # [B, 3]
    theta = AffineTransform(**normalization["theta"]).transform(theta_features(centers, config)[0])
    detector = config.target.detector_id if config.target.kind == "single_channel" else None
    labels = np.zeros((len(rows), width), dtype=np.int8)  # [B, N]
    for b, row in enumerate(rows):
        labels[b, : counts[b]] = _labels(cache, row, detector)
    phi = None
    if normalization["phi"] is not None:
        transform = AffineTransform(**normalization["phi"])
        phi = np.zeros((len(rows), width, len(transform.offset)), dtype=np.float32)
        for start, stop, values in phi_chunks(cache, rows, centers, config):
            values = transform.transform(values).astype(np.float32)  # [E, D]
            offset = 0
            for b in range(start, stop):
                phi[b, : counts[b]] = values[offset : offset + counts[b]]
                offset += counts[b]
    return StandardBatch(
        mode=InputMode.DESIGN_ONLY if phi is None else InputMode.FULL,
        theta=theta,
        phi=phi,
        labels=labels,
        event_counts=counts,
    )


def _labels(cache: EventCache, row: int, detector: int | None) -> np.ndarray:
    if detector is None:
        return cache.labels(row)
    hits = cache.hits(row)
    labels = np.zeros(int(cache.index["n_events"][row]), dtype=np.int8)
    labels[hits["row"][hits["detector"] == detector]] = 1
    return labels
