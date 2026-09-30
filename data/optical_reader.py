"""LH5 input validation. Importing this module does not require h5py."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from schemas.optical import ANOMALOUS_FILE, SourceConfig

NAME = re.compile(
    r"run\d+_x([+-]?\d+(?:\.\d+)?)_y([+-]?\d+(?:\.\d+)?)_z([+-]?\d+(?:\.\d+)?)\.stp\.lh5$"
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

    @property
    def n_events(self):
        return len(self.event_ids)


def read_run(path: Path, root: Path, *, fidelity: str | None = None) -> OpticalRun:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install resum-flex[optical-data] to read LH5 files") from exc
    try:
        relative = path.relative_to(root).as_posix() if fidelity is None else f"{fidelity}/{path.name}"
        fidelity = path.relative_to(root).parts[0] if fidelity is None else fidelity
        if fidelity not in ("hf", "lf"):
            raise ValueError("file must be inside hf/ or lf/")
        match = NAME.fullmatch(path.name)
        if not match:
            raise ValueError("filename must encode nominal x/y/z coordinates")
        with h5py.File(path, "r") as h:

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
            hit_ev = array("stp/optical/evtid", integer=True)
            det = array("stp/optical/det_uid", integer=True)
            ht = array("stp/optical/time", "ns")
            wave = array("stp/optical/wavelength", "nm")
            if any(len(a) != len(hit_ev) for a in [det, ht, wave]):
                raise ValueError("optical hit array length mismatch")
            if not np.isin(hit_ev, ev).all():
                raise ValueError("hit references unknown vertex event ID")
            positions = np.column_stack(pos).astype(float)
            momenta = np.column_stack(mom).astype(float)
        with path.open("rb") as f:
            fingerprint = hashlib.file_digest(f, "sha256").hexdigest()
        return OpticalRun(
            relative,
            fidelity,
            fingerprint,
            np.array(match.groups(), dtype=float),
            (positions.min(0) + positions.max(0)) / 2,
            ev,
            positions,
            momenta,
            time,
            n_part,
            hit_ev,
            det,
            ht,
            wave,
        )
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def read_optical_runs(config: SourceConfig) -> tuple[list[OpticalRun], list[dict]]:
    inputs = []
    if config.directory is not None:
        root = config.directory.resolve()
        for fidelity in ("hf", "lf"):
            inputs.extend((p, root, None) for p in sorted((root / fidelity).glob("*.stp.lh5")))
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
    exclusions = set(config.exclude_files) | {ANOMALOUS_FILE}
    runs, excluded = [], []
    seen_paths, seen_names = {}, {}
    for path, root, explicit_fidelity in inputs:
        name = path.relative_to(root).as_posix() if explicit_fidelity is None else f"{explicit_fidelity}/{path.name}"
        fidelity = name.split("/", 1)[0]
        resolved = path.resolve()
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
        else:
            runs.append(read_run(path, root, fidelity=explicit_fidelity))
    if not runs:
        raise ValueError("No included simulation files")
    return sorted(runs, key=lambda r: r.file), excluded
