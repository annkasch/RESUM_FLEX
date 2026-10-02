"""Filesystem manifests are authoritative; SQLite is a rebuildable local index."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def tree_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}


def snapshot_dataset(source, store):
    """Copy prepared arrays once; never hardlink mutable source files."""
    source, store = Path(source).resolve(), Path(store).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = [
        p
        for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix in {".npz", ".json", ".csv"}
    ]
    if not any(p.suffix == ".npz" for p in files):
        raise ValueError("Dataset must contain prepared .npz arrays")
    hashes = {str(p.relative_to(source)): file_hash(p) for p in files}
    identifier = digest(hashes)
    target = store / "datasets" / identifier
    manifest = {"schema_version": 1, "id": identifier, "files": hashes}
    if target.exists():
        verify_dataset(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=target.parent))
    try:
        for p in files:
            dest = temp / "data" / p.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        write_json(temp / "snapshot.json", manifest)
        verify_dataset(temp)
        try:
            temp.rename(target)
        except OSError:
            if not target.exists():
                raise
            verify_dataset(target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return target


def verify_dataset(path):
    path = Path(path)
    manifest = json.loads((path / "snapshot.json").read_text())
    actual = tree_hashes(path / "data")
    if actual != manifest["files"] or digest(actual) != manifest["id"]:
        raise ValueError(f"Dataset snapshot was modified: {path}")
    return manifest


def provenance(repo=None):
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo, stderr=subprocess.DEVNULL
            ).decode()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = {}
    for name in ("resum-flex", "numpy", "scipy", "torch", "GPy", "emukit", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Limit patch capture to implementation, not large executed notebooks or data.
    patch = git("diff", "HEAD", "--", "core", "schemas", "data", "viz", "pyproject.toml")
    untracked = git("ls-files", "--others", "--exclude-standard", "core", "schemas", "data", "viz")
    untracked_code = {}
    for name in (untracked or "").splitlines():
        p = Path(repo or ".") / name
        if p.suffix == ".py":
            untracked_code[name] = p.read_text()
    import torch

    return {
        "captured_at": now(),
        "git_commit": (git("rev-parse", "HEAD") or "").strip() or None,
        "git_status": git("status", "--porcelain"),
        "code_patch": patch,
        "untracked_code": untracked_code,
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
        "thread_environment": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, config, *, parents=(), kind="experiment"):
        identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
        path = self.root / "runs" / identifier
        path.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "id": identifier,
            "kind": kind,
            "name": config.get("name", identifier),
            "hypothesis": config.get("hypothesis", ""),
            "tags": config.get("tags", []),
            "parents": list(parents),
            "status": "pending",
            "created_at": now(),
            "updated_at": now(),
            "config": config,
            "path": str(path),
        }
        write_json(path / "run.json", manifest)
        self.index(manifest)
        return path

    def transition(self, path, status, **details):
        path = Path(path)
        manifest = json.loads((path / "run.json").read_text())
        allowed = {"pending": {"running", "failed"}, "running": {"completed", "failed"}}
        if status not in allowed.get(manifest["status"], set()):
            raise ValueError(f"Immutable or invalid transition: {manifest['status']} -> {status}")
        manifest.update(details, status=status, updated_at=now())
        if status == "completed":
            manifest["artifacts"] = {k: v for k, v in tree_hashes(path).items() if k != "run.json"}
        write_json(path / "run.json", manifest)
        self.index(manifest)
        return manifest

    def connect(self):
        con = sqlite3.connect(self.root / "catalog.sqlite", timeout=30)
        con.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        return con

    def index(self, record):
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?)", (record["id"], json.dumps(record))
            )

    def rebuild(self):
        records = [
            json.loads(p.read_text()) for p in sorted((self.root / "runs").glob("*/run.json"))
        ]
        with self.connect() as con:
            con.execute("DELETE FROM runs")
            con.executemany(
                "INSERT INTO runs VALUES (?, ?)", [(r["id"], json.dumps(r)) for r in records]
            )
        return records

    def list(self, *, status=None, tag=None):
        with self.connect() as con:
            records = [
                json.loads(row[0]) for row in con.execute("SELECT record FROM runs ORDER BY id")
            ]
        return [
            r
            for r in records
            if (status is None or r["status"] == status) and (tag is None or tag in r["tags"])
        ]

    def inspect(self, identifier, *, verify=True):
        path = self.root / "runs" / identifier
        r = json.loads((path / "run.json").read_text())
        if verify and r["status"] == "completed":
            actual = {k: v for k, v in tree_hashes(path).items() if k != "run.json"}
            if actual != r["artifacts"]:
                raise ValueError(f"Completed run artifacts were modified: {identifier}")
        return r
