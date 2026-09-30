"""Stable group-aware splits, independent of feature construction and outcomes."""

import hashlib
import json
from pathlib import Path

import numpy as np

from data.optical_features import center
from schemas.optical import OpticalDataConfig

SPLITS = ("train", "validation", "test")
FIDELITIES = ("lf", "hf")
TOLERANCE = 1e-6


def voxel_groups(runs, config):
    parent = list(range(len(runs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    nominal = {}
    for i, r in enumerate(runs):
        key = tuple(r.nominal_center)
        if key in nominal:
            union(i, nominal[key])
        nominal[key] = i
    if config.theta.coordinates == "cylindrical" and not config.theta.cylindrical.include_azimuth:
        coords = np.array([[np.hypot(*center(r, config)[:2]), center(r, config)[2]] for r in runs])
        for i in range(len(runs)):
            for j in range(i):
                if np.all(np.abs(coords[i] - coords[j]) <= TOLERANCE):
                    union(i, j)
    groups = {}
    for i, run in enumerate(runs):
        groups.setdefault(find(i), []).append(run)
    return sorted(groups.values(), key=lambda rs: min(r.file for r in rs))


def integer_targets(n, fractions):
    raw = n * np.asarray(fractions)
    targets = np.floor(raw).astype(int)
    order = np.argsort(-(raw - targets), kind="stable")
    targets[order[: n - int(targets.sum())]] += 1
    return targets


def assign_splits(runs, config: OpticalDataConfig, *, update=False):
    """Update preserves assignments of surviving files; conflicting groups fail."""
    path = config.split.manifest
    groups = voxel_groups(runs, config)
    settings = {"fractions": list(config.split.fractions), "seed": config.split.seed}
    previous = {}
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("version") != 1:
            raise ValueError("Unsupported split manifest version")
        previous = {r["file"]: r for r in old["files"]}
        if len(previous) != len(old["files"]):
            raise ValueError("Duplicate files in split manifest")
        if old["settings"] != settings and not update:
            raise ValueError(
                "Split settings changed; use explicit manifest update or a new manifest"
            )
        current = {r.file: r for r in runs}
        changed = set(previous) != set(current) or any(
            previous[name]["fingerprint"] != run.fingerprint
            or previous[name]["fidelity"] != run.fidelity
            or previous[name]["nominal_center"] != run.nominal_center.tolist()
            for name, run in current.items()
            if name in previous
        )
        if changed and not update:
            raise ValueError("Source files changed; use explicit manifest update")
    counts = np.zeros((3, 2), dtype=int)
    targets = np.column_stack(
        [
            integer_targets(sum(r.fidelity == f for r in runs), config.split.fractions)
            for f in FIDELITIES
        ]
    )
    forced = {}
    if config.split.lf_train_only:
        forced.update({r.file: "train" for r in runs if r.fidelity == "lf"})
        targets[:, 0] = [sum(r.fidelity == "lf" for r in runs), 0, 0]
    if config.split.hf_train_count is not None:
        hf = sorted(r.file for r in runs if r.fidelity == "hf")
        n_train = config.split.hf_train_count
        if n_train >= len(hf):
            raise ValueError("hf_train_count must leave at least one HF validation voxel")
        selected = set(np.random.default_rng(config.split.seed).choice(hf, n_train, replace=False))
        forced.update({name: "train" if name in selected else "validation" for name in hf})
        targets[:, 1] = [n_train, len(hf) - n_train, 0]
    assignments, pending = {}, []
    for group in groups:
        locked = {previous[r.file]["split"] for r in group if r.file in previous}
        if not locked.issubset(SPLITS) or len(locked) > 1:
            raise ValueError("Equivalent voxel inputs cross splits; use a new versioned manifest")
        requested = {forced[r.file] for r in group if r.file in forced}
        if len(requested) > 1:
            raise ValueError("Explicit LF/HF split policy conflicts within a voxel group")
        if requested and locked and requested != locked and not update:
            raise ValueError("Explicit split policy changed; use a new manifest or explicit update")
        if requested or locked:
            split = next(iter(requested or locked))
            for r in group:
                assignments[r.file] = split
                counts[SPLITS.index(split), FIDELITIES.index(r.fidelity)] += 1
        else:
            pending.append(group)
    order = np.random.default_rng(config.split.seed).permutation(len(pending))
    for i in order:
        group = pending[i]
        increment = np.array([sum(r.fidelity == f for r in group) for f in FIDELITIES])
        scores = []
        for s in range(3):
            candidate = counts.copy()
            candidate[s] += increment
            scores.append(np.square(candidate - targets).sum())
        s = int(np.argmin(scores))
        counts[s] += increment
        for r in group:
            assignments[r.file] = SPLITS[s]
    records = []
    for group in groups:
        identities = sorted(set(tuple(r.nominal_center.tolist()) for r in group))
        group_id = hashlib.sha256(json.dumps(identities).encode()).hexdigest()[:16]
        for r in group:
            records.append(
                {
                    "file": r.file,
                    "fidelity": r.fidelity,
                    "group_id": group_id,
                    "nominal_center": r.nominal_center.tolist(),
                    "fingerprint": r.fingerprint,
                    "split": assignments[r.file],
                }
            )
    payload = {
        "version": 1,
        "settings": settings,
        "policy": {"lf_train_only": config.split.lf_train_only, "hf_train_count": config.split.hf_train_count},
        "files": sorted(records, key=lambda r: r["file"]),
        "targets": targets.tolist(),
        "actual_counts": counts.tolist(),
        "split_order": list(SPLITS),
        "fidelity_order": list(FIDELITIES),
        "overlapping_groups": 0,
    }
    return assignments, payload


def save_manifest(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)
