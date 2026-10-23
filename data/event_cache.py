"""Shared float32 copy of raw optical LH5 files, read once and reused by every data config.

Layout of a cache directory::

    cache.json        version, folder list, shard list
    index.npy         one row per cached file (INDEX_DTYPE)
    shards/<name>/    events.npy  float32 [n_events, 6]  vertex x/y/z (m), momentum px/py/pz (MeV)
                      labels.npy  uint8   [n_events]     1 if any detector saw the event
                      hits.npy    HIT_DTYPE [n_hits]     (event row in file, detector, hits)

Each file's events are one contiguous block of rows in one shard (``first``, ``n_events``),
in the LH5 vertex order; its hit list is the block ``first_hit``, ``n_hits`` of the shard's
hits, one row per (event, detector) pair. Event IDs, vertex times, particle counts, hit times
and wavelengths are validated while reading but not stored.

New or changed raw files (name, size, modification time) are read by parallel worker
processes into new shards; a changed file's old row is marked invalid. ``compact`` rewrites
the valid rows into fewer shards. Only one process may update or compact a cache at a time,
and compacting must not run while another process reads the cache.

Command line::

    python -m data.event_cache update CONFIG.yaml   # cache the inputs of a data config
    python -m data.event_cache compact CACHE_DIR
"""

import hashlib
import io
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from data.optical_reader import nominal_center, read_arrays, select_inputs
from schemas.optical import SourceConfig

VERSION = 1
NAME_BYTES = 96
INDEX_DTYPE = np.dtype(
    [
        ("folder", "i4"),  # position in cache.json "folders" (resolved folder path)
        ("name", f"S{NAME_BYTES}"),  # file name inside the folder
        ("size", "i8"),
        ("mtime_ns", "i8"),
        ("sha256", "u1", (32,)),
        ("shard", "i4"),  # position in cache.json "shards"
        ("first", "i8"),  # first event row in the shard
        ("n_events", "i4"),
        ("first_hit", "i8"),  # first row in the shard's hits
        ("n_hits", "i4"),
        ("nominal_center", "f8", (3,)),
        ("measured_center", "f8", (3,)),  # midpoint of the vertex range
        ("max_n_part", "i4"),  # largest number of primaries in one event
        ("valid", "?"),  # False once the file changed and was cached again
    ]
)
HIT_DTYPE = np.dtype([("row", "i4"), ("detector", "i4"), ("hits", "i4")])
FILES_PER_SHARD = 20_000
MIN_FILES_PER_TASK = 50  # starting a worker process costs about as much as 50 file reads
PARTS = ("events", "labels", "hits")


def default_workers() -> int:
    """CPUs available to this process (the Slurm allocation on compute nodes)."""
    return len(os.sched_getaffinity(0))


def _read_file(path: Path):
    """Validated cache content of one LH5 file; the file is read from disk exactly once."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install resum-flex[optical-data] to read LH5 files") from exc
    data = path.read_bytes()
    digest = np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8)
    with h5py.File(io.BytesIO(data), "r") as h:
        a = read_arrays(h)
    ids = a["event_ids"]
    order = np.argsort(ids)
    rows = order[np.searchsorted(ids, a["hit_event_ids"], sorter=order)]
    pairs, counts = np.unique(
        np.column_stack([rows, a["detector_ids"]]).astype(np.int64), axis=0, return_counts=True
    )
    hits = np.empty(len(pairs), dtype=HIT_DTYPE)
    hits["row"], hits["detector"], hits["hits"] = pairs[:, 0], pairs[:, 1], counts
    labels = np.zeros(len(ids), dtype=np.uint8)
    labels[rows] = 1
    events = np.column_stack([a["positions"], a["momenta"]]).astype(np.float32)
    measured = (a["positions"].min(0).astype(float) + a["positions"].max(0)) / 2
    return events, labels, hits, digest, measured, int(a["n_part"].max())


def _write_shard(directory: Path, events, labels, hits):
    """Write a complete shard, so a crash never leaves a partial one under its final name."""
    temp = directory.with_name(f".tmp-{directory.name}")
    temp.mkdir(parents=True)
    np.save(temp / "events.npy", events)
    np.save(temp / "labels.npy", labels)
    np.save(temp / "hits.npy", hits)
    temp.rename(directory)


def _build_shard(task):
    """Worker: read the given files into one new shard; return their index fields."""
    directory, files = task
    events, labels, hits, records = [], [], [], []
    first = first_hit = 0
    for path in map(Path, files):
        try:
            center = nominal_center(path.name)
            e, lab, h, digest, measured, n_part = _read_file(path)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        events.append(e)
        labels.append(lab)
        hits.append(h)
        records.append(
            (str(path), digest, first, len(e), first_hit, len(h), center, measured, n_part)
        )
        first += len(e)
        first_hit += len(h)
    _write_shard(
        Path(directory), np.concatenate(events), np.concatenate(labels), np.concatenate(hits)
    )
    return records


def _identity(path: str) -> tuple[int, int]:
    info = os.stat(path)
    return info.st_size, info.st_mtime_ns


def identities(paths) -> list[tuple[int, int]]:
    """(size, mtime_ns) per path, fetched in parallel; serial stat calls are slow on CFS."""
    with ThreadPoolExecutor(max_workers=32) as pool:
        return list(pool.map(_identity, map(str, paths), chunksize=256))


class EventCache:
    """Read access to a cache directory; ``update`` adds new or changed raw files."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self._load()

    def _load(self):
        meta = self.directory / "cache.json"
        if meta.is_file():
            info = json.loads(meta.read_text())
            if info.get("version") != VERSION:
                raise ValueError(f"Unsupported event cache version in {self.directory}")
            self.folders, self.shards = info["folders"], info["shards"]
            self.index = np.load(self.directory / "index.npy")
        else:
            self.folders, self.shards = [], []
            self.index = np.empty(0, dtype=INDEX_DTYPE)
        self._mapped, self._lookup = {}, None

    # --- lookup -----------------------------------------------------------------------------
    def lookup(self) -> dict[tuple[str, str], int]:
        """Valid rows by (resolved folder, file name)."""
        if self._lookup is None:
            valid = np.flatnonzero(self.index["valid"])
            folders, names = self.index["folder"][valid], self.index["name"][valid]
            self._lookup = {
                (self.folders[f], n.decode()): int(r)
                for r, f, n in zip(valid, folders, names, strict=True)
            }
        return self._lookup

    def locate(self, paths, sha256=None) -> np.ndarray:
        """Index rows of resolved file paths; with ``sha256`` ([n, 32] uint8) check contents.

        Files on disk are not checked; callers compare raw-file stamps beforehand.
        """
        table = self.lookup()
        rows = np.empty(len(paths), dtype=np.int64)
        for i, path in enumerate(map(Path, paths)):
            row = table.get((str(path.parent), path.name))
            if row is None:
                raise ValueError(f"{path} is missing from the event cache {self.directory}")
            rows[i] = row
        if sha256 is not None and not np.array_equal(self.index["sha256"][rows], sha256):
            raise ValueError(f"Event cache {self.directory} holds different file contents")
        return rows

    def path(self, row: int) -> str:
        r = self.index[row]
        return os.path.join(self.folders[r["folder"]], r["name"].decode())

    def shard(self, number: int, part: str) -> np.ndarray:
        """Memory-mapped shard array ``part`` (events, labels or hits)."""
        key = number, part
        if key not in self._mapped:
            path = self.directory / "shards" / self.shards[number] / f"{part}.npy"
            self._mapped[key] = np.load(path, mmap_mode="r")
        return self._mapped[key]

    def events(self, row: int) -> np.ndarray:  # [n_events, 6] float32
        r = self.index[row]
        return self.shard(r["shard"], "events")[r["first"] : r["first"] + r["n_events"]]

    def labels(self, row: int) -> np.ndarray:  # [n_events] uint8, any detector
        r = self.index[row]
        return self.shard(r["shard"], "labels")[r["first"] : r["first"] + r["n_events"]]

    def hits(self, row: int) -> np.ndarray:  # [n_hits] HIT_DTYPE
        r = self.index[row]
        return self.shard(r["shard"], "hits")[r["first_hit"] : r["first_hit"] + r["n_hits"]]

    # --- update -----------------------------------------------------------------------------
    @contextmanager
    def _locked(self):
        """Exclusive write access across processes and nodes; reloads the current state.

        A directory lock works on every file system. A crashed writer leaves it behind;
        remove it by hand once no other process is writing.
        """
        lock = self.directory / ".lock"
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            raise RuntimeError(
                f"Event cache {self.directory} is locked by another writer; if none is "
                f"running, remove {lock}"
            ) from None
        try:
            self._load()
            yield
        finally:
            lock.rmdir()

    def update(
        self,
        paths,
        *,
        workers: int | None = None,
        files_per_shard: int = FILES_PER_SHARD,
        min_files_per_task: int = MIN_FILES_PER_TASK,
    ) -> int:
        """Cache every resolved path that is new or changed; return how many were read."""
        paths = [Path(p) for p in paths]
        found = identities(paths)
        if self._outdated(paths, found):
            with self._locked():
                return self._update(paths, found, workers, files_per_shard, min_files_per_task)
        return 0

    def _outdated(self, paths, found) -> list:
        table = self.lookup()
        stored = self.index[["size", "mtime_ns"]]
        todo = []
        for path, identity in zip(paths, found, strict=True):
            row = table.get((str(path.parent), path.name))
            if row is None or tuple(stored[row].tolist()) != identity:
                if len(path.name.encode()) > NAME_BYTES:
                    raise ValueError(f"File name longer than {NAME_BYTES} bytes: {path.name}")
                todo.append((path, identity))
        return todo

    def _update(self, paths, found, workers, files_per_shard, min_files_per_task) -> int:
        todo = self._outdated(paths, found)  # again: another writer may have added files
        if not todo:
            return 0
        table = self.lookup()
        workers = workers or default_workers()
        size = max(1, min(files_per_shard, max(min_files_per_task, -(-len(todo) // workers))))
        stamp = f"{os.getpid()}-{os.urandom(4).hex()}"
        shards = self.directory / "shards"
        shards.mkdir(parents=True, exist_ok=True)
        names = [f"{stamp}-{k:05d}" for k in range(-(-len(todo) // size))]
        tasks = [
            (shards / name, [str(p) for p, _ in todo[k * size : (k + 1) * size]])
            for k, name in enumerate(names)
        ]
        if workers == 1 or len(tasks) == 1:
            results = list(map(_build_shard, tasks))
        else:
            context = get_context("spawn")
            with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=context) as pool:
                results = list(pool.map(_build_shard, tasks))
        folders = {f: i for i, f in enumerate(self.folders)}
        added = []
        for name, records in zip(names, results, strict=True):
            self.shards.append(name)
            for path, digest, first, n, first_hit, n_hits, center, measured, n_part in records:
                path = Path(path)
                folder = folders.setdefault(str(path.parent), len(folders))
                added.append(
                    (
                        folder,
                        path.name.encode(),
                        0,
                        0,
                        digest,
                        len(self.shards) - 1,
                        first,
                        n,
                        first_hit,
                        n_hits,
                        center,
                        measured,
                        n_part,
                        True,
                    )
                )
        added = np.array(added, dtype=INDEX_DTYPE)
        added["size"], added["mtime_ns"] = np.array([identity for _, identity in todo]).T
        for path, _ in todo:
            old = table.get((str(path.parent), path.name))
            if old is not None:
                self.index["valid"][old] = False
        self.folders = list(folders)
        self.index = np.concatenate([self.index, added])
        self._lookup = None
        self._save()
        return len(todo)

    def compact(self, *, files_per_shard: int = FILES_PER_SHARD) -> int:
        """Copy valid rows into new shards of ``files_per_shard`` files; return the shard count.

        Row numbers change; views refer to files by path and content hash, not by row.
        Shards left behind by failed updates are removed too.
        """
        with self._locked():
            return self._compact(files_per_shard)

    def _compact(self, files_per_shard: int) -> int:
        valid = np.flatnonzero(self.index["valid"])
        order = valid[np.lexsort((self.index["first"][valid], self.index["shard"][valid]))]
        stamp = f"compact-{os.getpid()}-{os.urandom(4).hex()}"
        new_shards = []
        index = self.index[order].copy()
        for k, start in enumerate(range(0, len(order), files_per_shard)):
            rows = order[start : start + files_per_shard]
            parts = {part: [] for part in PARTS}
            first = first_hit = 0
            for i, row in enumerate(rows, start):
                parts["events"].append(self.events(row))
                parts["labels"].append(self.labels(row))
                parts["hits"].append(self.hits(row))
                index[i]["shard"], index[i]["first"], index[i]["first_hit"] = k, first, first_hit
                first += index[i]["n_events"]
                first_hit += index[i]["n_hits"]
            name = f"{stamp}-{k:05d}"
            _write_shard(
                self.directory / "shards" / name, *(np.concatenate(parts[p]) for p in PARTS)
            )
            new_shards.append(name)
        self._mapped, self._lookup = {}, None
        self.shards, self.index = new_shards, index
        self._save()
        for path in (self.directory / "shards").iterdir():
            if path.name not in new_shards:
                shutil.rmtree(path)
        return len(new_shards)

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        # Lists first: extra folders or shards are harmless if writing the index is interrupted.
        meta = {"version": VERSION, "folders": self.folders, "shards": self.shards}
        temp = self.directory / "cache.json.tmp"
        temp.write_text(json.dumps(meta, indent=1) + "\n")
        temp.replace(self.directory / "cache.json")
        temp = self.directory / "index.tmp.npy"
        np.save(temp, self.index)
        temp.replace(self.directory / "index.npy")


@dataclass(eq=False)
class CachedRun:
    """One cached simulation file with the attributes splits, features and reports use."""

    cache: EventCache
    row: int
    file: str  # fidelity/filename, as in the split manifest
    fidelity: str
    source: int | None = None  # index into SourceConfig.sources (role-based layout only)

    @property
    def path(self) -> str:
        return self.cache.path(self.row)

    @property
    def fingerprint(self) -> str:
        return self.cache.index["sha256"][self.row].tobytes().hex()

    @property
    def nominal_center(self) -> np.ndarray:
        return self.cache.index["nominal_center"][self.row]

    @property
    def measured_center(self) -> np.ndarray:
        return self.cache.index["measured_center"][self.row]

    @property
    def n_events(self) -> int:
        return int(self.cache.index["n_events"][self.row])

    @property
    def positions(self) -> np.ndarray:  # [n_events, 3] float32, m
        return self.cache.events(self.row)[:, :3]

    @property
    def momenta(self) -> np.ndarray:  # [n_events, 3] float32, MeV
        return self.cache.events(self.row)[:, 3:]

    @property
    def max_n_part(self) -> int:
        """Largest number of primaries in one event of the file."""
        return int(self.cache.index["max_n_part"][self.row])

    @property
    def event_ids(self) -> np.ndarray:
        """Event rows; hit_event_ids refer to these, not to the LH5 event IDs."""
        return np.arange(self.n_events)

    @property
    def hit_event_ids(self) -> np.ndarray:
        hits = self.cache.hits(self.row)
        return np.repeat(hits["row"], hits["hits"])

    @property
    def detector_ids(self) -> np.ndarray:
        hits = self.cache.hits(self.row)
        return np.repeat(hits["detector"], hits["hits"])


def cached_runs(source: SourceConfig, cache: EventCache, *, workers: int | None = None):
    """Runs of the configured inputs, sorted by name, after caching new or changed files."""
    included, excluded = select_inputs(source)
    paths = [resolved for *_, resolved in included]
    read = cache.update(paths, workers=workers)
    print(
        f"Event cache {cache.directory}: {read} files read, {len(paths) - read} reused", flush=True
    )
    rows = cache.locate(paths)
    runs = [
        CachedRun(cache, int(row), name, name.split("/", 1)[0], source_index)
        for row, (_, _, _, name, source_index, _) in zip(rows, included, strict=True)
    ]
    return sorted(runs, key=lambda r: r.file), excluded


def main(argv=None):
    import argparse

    from schemas.optical import load_optical_config

    parser = argparse.ArgumentParser(prog="python -m data.event_cache")
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser("update", help="cache the raw inputs of a data config")
    update.add_argument("config", type=Path)
    update.add_argument("--workers", type=int)
    compact = commands.add_parser("compact", help="merge shards and drop outdated rows")
    compact.add_argument("directory", type=Path)
    compact.add_argument("--files-per-shard", type=int, default=FILES_PER_SHARD)
    args = parser.parse_args(argv)
    if args.command == "update":
        config = load_optical_config(args.config)
        cached_runs(config.source, EventCache(config.event_cache), workers=args.workers)
    else:
        n = EventCache(args.directory).compact(files_per_shard=args.files_per_shard)
        print(f"Compacted {args.directory} into {n} shards")


if __name__ == "__main__":
    main()
