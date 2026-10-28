"""LH5 input validation. Importing this module does not require h5py."""

import errno
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from schemas.optical import ANOMALOUS_FILE, SourceConfig

# An optional `_v{voxel index}` segment (`run00001_v00003913_x+...`) is accepted and ignored.
NAME = re.compile(
    r"run\d+(?:_v\d+)?_x([+-]?\d+(?:\.\d+)?)_y([+-]?\d+(?:\.\d+)?)"
    r"_z([+-]?\d+(?:\.\d+)?)\.stp\.lh5$"
)


@dataclass
class OpticalRun:
    file: str
    fidelity: str
    fingerprint: str
    nominal_center: np.ndarray
    measured_center: np.ndarray
    event_ids: np.ndarray
    positions: np.ndarray
    momenta: np.ndarray
    vertex_time: np.ndarray
    n_part: np.ndarray
    hit_event_ids: np.ndarray
    detector_ids: np.ndarray
    hit_time: np.ndarray
    hit_wavelength: np.ndarray
    source: int | None = None  # index into SourceConfig.sources (role-based layout only)

    @property
    def n_events(self):
        return len(self.event_ids)


def nominal_center(name: str) -> np.ndarray:
    """Voxel center encoded in a simulation filename."""
    match = NAME.fullmatch(name)
    if not match:
        raise ValueError("filename must encode nominal x/y/z coordinates")
    return np.array(match.groups(), dtype=float)


def read_arrays(h, *, require_hits: bool = True) -> dict:
    """Validated vertex and optical-hit arrays of one open LH5 file (h5py.File).

    With ``require_hits=False`` a file without the ``stp/optical`` group is accepted:
    ``labeled`` is False and the hit arrays are empty. Files with hits are always checked.
    """

    def array(key, unit=None, integer=False):
        d = h[key]
        a = d[:]
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError(f"{key}: expected finite 1-D array")
        if integer and not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"{key}: expected integer values")
        if unit and d.attrs.get("units") not in (unit, unit.encode()):
            raise ValueError(f"{key}: expected units {unit}")
        return a

    n_raw = h["number_of_simulated_events"][()]
    if np.ndim(n_raw) != 0 or not np.isfinite(n_raw) or n_raw <= 0:
        raise ValueError("invalid simulated event count")
    n = int(n_raw)
    if n != n_raw:
        raise ValueError("non-integral simulated event count")
    ev = array("vtx/evtid", integer=True)
    if len(ev) != n or len(np.unique(ev)) != n:
        raise ValueError("vertex IDs must be unique and match simulated event count")
    pos = [array("vtx/" + a + "loc", "m") for a in "xyz"]
    mom = [array("vtx/p" + a, "MeV") for a in "xyz"]
    time = array("vtx/time", "ns")
    n_part = array("vtx/n_part", integer=True)
    if any(len(a) != n for a in [*pos, *mom, time, n_part]):
        raise ValueError("vertex array length mismatch")
    if np.any(n_part < 1):
        raise ValueError("n_part must be positive")
    vertex = dict(
        event_ids=ev,
        positions=np.column_stack(pos),
        momenta=np.column_stack(mom),
        vertex_time=time,
        n_part=n_part,
    )
    if "stp/optical" not in h:
        if require_hits:
            raise ValueError("no optical hit data (stp/optical)")
        empty_int, empty = np.zeros(0, dtype=np.int64), np.zeros(0)
        return dict(
            **vertex,
            labeled=False,
            hit_event_ids=empty_int,
            detector_ids=empty_int,
            hit_time=empty,
            hit_wavelength=empty,
        )
    hit_ev = array("stp/optical/evtid", integer=True)
    det = array("stp/optical/det_uid", integer=True)
    ht = array("stp/optical/time", "ns")
    wave = array("stp/optical/wavelength", "nm")
    if any(len(a) != len(hit_ev) for a in [det, ht, wave]):
        raise ValueError("optical hit array length mismatch")
    if not np.isin(hit_ev, ev).all():
        raise ValueError("hit references unknown vertex event ID")
    return dict(
        **vertex,
        labeled=True,
        hit_event_ids=hit_ev,
        detector_ids=det,
        hit_time=ht,
        hit_wavelength=wave,
    )


def read_run(path: Path, root: Path, *, fidelity: str | None = None) -> OpticalRun:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install resum-flex[optical-data] to read LH5 files") from exc
    try:
        relative = (
            path.relative_to(root).as_posix() if fidelity is None else f"{fidelity}/{path.name}"
        )
        fidelity = path.relative_to(root).parts[0] if fidelity is None else fidelity
        if fidelity not in ("hf", "lf"):
            raise ValueError("file must be inside hf/ or lf/")
        center = nominal_center(path.name)
        with h5py.File(path, "r") as h:
            a = read_arrays(h)
        positions, momenta = a["positions"].astype(float), a["momenta"].astype(float)
        with path.open("rb") as f:
            fingerprint = hashlib.file_digest(f, "sha256").hexdigest()
        return OpticalRun(
            relative,
            fidelity,
            fingerprint,
            center,
            (positions.min(0) + positions.max(0)) / 2,
            a["event_ids"],
            positions,
            momenta,
            a["vertex_time"],
            a["n_part"],
            a["hit_event_ids"],
            a["detector_ids"],
            a["hit_time"],
            a["hit_wavelength"],
        )
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def list_source_files(config: SourceConfig) -> list[tuple[Path, Path, str | None]]:
    """(file, root folder, explicit fidelity) for every configured input, without reading it."""
    inputs = []
    if config.directory is not None:
        root = config.directory.resolve()
        for fidelity in ("hf", "lf"):
            inputs.extend((p, root, None) for p in sorted((root / fidelity).glob("*.stp.lh5")))
    elif config.sources:
        for folder in (s.path for s in config.sources):
            folder = folder.expanduser().resolve()
            if not folder.is_dir():
                raise FileNotFoundError(f"Optical input folder does not exist: {folder}")
            paths = sorted(folder.glob("*.stp.lh5"))
            if not paths:
                raise ValueError(f"No simulation files found in {folder}")
            inputs.extend((p, folder, "lf") for p in paths)
    else:
        for fidelity, folders in sorted(config.directories.items()):
            for folder in folders:
                folder = folder.expanduser().resolve()
                if not folder.is_dir():
                    raise FileNotFoundError(f"Optical input folder does not exist: {folder}")
                paths = sorted(folder.glob("*.stp.lh5"))
                if not paths:
                    raise ValueError(f"No simulation files found in {folder}")
                inputs.extend((p, folder, fidelity) for p in paths)
    if not inputs:
        raise ValueError("No simulation files found in configured folders")
    return inputs


def _file_identity(path: str) -> str:
    info = os.stat(path)
    return f"{os.path.basename(path)}:{info.st_size}:{info.st_mtime_ns}"


def source_stamp(config: SourceConfig) -> list[list]:
    """Cheap identity of the raw inputs without opening any file.

    Per folder: file count and a hash over every file's name, size and modification time.
    Adding, removing, renaming or rewriting a file changes it. Metadata is fetched in
    parallel, because one-by-one stat calls are slow on network file systems.
    """
    if config.directory is not None:
        folders = [config.directory / f for f in ("hf", "lf") if (config.directory / f).is_dir()]
    elif config.sources:
        folders = [s.path for s in config.sources]
    else:
        folders = [f for _, fs in sorted(config.directories.items()) for f in fs]
    stamp = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        for folder in folders:
            folder = folder.expanduser().resolve()
            paths = sorted(e.path for e in os.scandir(folder) if e.name.endswith(".stp.lh5"))
            identities = "\n".join(pool.map(_file_identity, paths))
            stamp.append([str(folder), len(paths), hashlib.sha256(identities.encode()).hexdigest()])
    return stamp


def real_paths(paths) -> list[Path]:
    """Paths with every symlink followed, like ``Path.resolve``, for many files at once.

    ``Path.resolve`` checks every path component (~20 metadata calls per file, slow on CFS).
    Here each file costs one ``readlink`` per link, run in parallel, and every folder is
    resolved only once.
    """
    folders = {}

    def folder(path: str) -> str:
        if path not in folders:
            folders[path] = os.path.realpath(path)
        return folders[path]

    def one(path) -> Path:
        path = os.path.join(folder(os.path.dirname(os.path.abspath(path))), os.path.basename(path))
        for _ in range(40):
            try:
                target = os.readlink(path)
            except OSError as exc:
                if exc.errno != errno.EINVAL:  # EINVAL: not a link
                    raise
                return Path(path)
            target = os.path.join(os.path.dirname(path), target)
            path = os.path.join(folder(os.path.dirname(target)), os.path.basename(target))
        raise OSError(f"Too many levels of symbolic links: {path}")

    with ThreadPoolExecutor(max_workers=32) as pool:
        return list(pool.map(one, paths, chunksize=256))


def select_inputs(config: SourceConfig) -> tuple[list[tuple], list[dict]]:
    """Included inputs as (path, root, explicit fidelity, name, source index, resolved path),
    plus exclusions.

    ``name`` is ``fidelity/filename``; duplicates through aliases are dropped, conflicting
    duplicates fail. Nothing is opened.
    """
    inputs = list_source_files(config)
    resolved_paths = real_paths([path for path, *_ in inputs])
    sources = {s.path.expanduser().resolve(): i for i, s in enumerate(config.sources or [])}
    exclusions = set(config.exclude_files) | {ANOMALOUS_FILE}
    included, excluded = [], []
    seen_paths, seen_names = {}, {}
    for (path, root, explicit_fidelity), resolved in zip(inputs, resolved_paths, strict=True):
        name = (
            path.relative_to(root).as_posix()
            if explicit_fidelity is None
            else f"{explicit_fidelity}/{path.name}"
        )
        fidelity = name.split("/", 1)[0]
        if resolved in seen_paths:
            if seen_paths[resolved] != fidelity:
                raise ValueError(f"Same source file assigned to LF and HF: {path}")
            continue
        if name in seen_names:
            raise ValueError(f"Ambiguous simulation filename {name} in multiple folders")
        seen_paths[resolved], seen_names[name] = fidelity, resolved
        if name in exclusions:
            excluded.append(
                {
                    "file": name,
                    "reason": "known anomaly" if name == ANOMALOUS_FILE else "configured exclusion",
                }
            )
            continue
        source = sources[root] if config.sources else None
        included.append((path, root, explicit_fidelity, name, source, resolved))
    if not included:
        raise ValueError("No included simulation files")
    return included, excluded


def read_optical_runs(config: SourceConfig) -> tuple[list[OpticalRun], list[dict]]:
    included, excluded = select_inputs(config)
    runs = []
    for path, root, explicit_fidelity, _, source, _ in included:
        run = read_run(path, root, fidelity=explicit_fidelity)
        run.source = source
        runs.append(run)
    return sorted(runs, key=lambda r: r.file), excluded
