"""Pipeline smoke check: every CLI stage once on tiny seeded synthetic data.

A pipeline check only (exit codes and output files), no model conclusions. Runs in a fresh
temporary directory, under a minute, safe on a login node (one thread, CPU):

    train (mlp + last layer) -> predict -> mfgp -> binomial (event-model covariate),
    stream (stream MLP on an in-memory event source, scored on the design-point data),
    residual (on the train and stream runs; only with the lgbm extra),
    within_point (on the train run)

Each stage runs the real CLI (``python -m core.surrogates ...``) in a subprocess; independent
stages run side by side (at most 3 single-thread processes). Prints OK and the wall time per
stage; exits 1 at the first failing stage (its output is shown, the directory kept).

    python scripts/smoke.py [--keep] [--workdir DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # `python scripts/smoke.py` without PYTHONPATH

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from core.surrogates.datasets import NpzBatches  # noqa: E402
from data.pseudo_generator import for_scenario  # noqa: E402

SCENARIO = "S4"  # FULL mode, 2-D theta and 2-D phi: catches shape bugs S1 would hide
DIM_THETA, DIM_PHI = 2, 2
# (partition, fidelity, design points, events per design point)
SETS = [
    ("train", "lf", 12, 40),
    ("validation", "lf", 6, 40),
    ("validation", "hf", 6, 120),
    ("gp_train", "lf", 10, 40),
    ("gp_train", "hf", 5, 120),
    ("gp_validation", "hf", 6, 120),
]
# Partitions the event model predicts for the GPs.
PREDICTED = [("gp_train", "lf"), ("gp_train", "hf"), ("gp_validation", "hf")]
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


class SmokeNpz(NpzBatches):
    """Npz batches with an identity affine normalization (inference requires one)."""

    def normalization(self):
        return {
            "theta": {
                "method": "standard",
                "offset": [0.0] * DIM_THETA,
                "scale": [1.0] * DIM_THETA,
            },
            "phi": {"method": "standard", "offset": [0.0] * DIM_PHI, "scale": [1.0] * DIM_PHI},
        }


class SmokeEventSource:
    """In-memory event source over ``<store>/<split>.npy`` records (core.event_stream)."""

    def __init__(self, store, split, *, seed=0, rank=0, world_size=1):
        self.records = np.load(Path(store) / f"{split}.npy")
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.n_events = len(self.records)
        self.n_positive = int((self.records[:, -1].view(np.int32) != 0).sum())

    def normalization(self):
        width = DIM_THETA + DIM_PHI  # same units as the design-point data (identity)
        return {"offset": [0.0] * width, "scale": [1.0] * width}

    def n_batches(self, batch_size, pass_index, *, drop_last=False):
        n = len(self.records) // self.world_size
        return n // batch_size if drop_last else -(-n // batch_size)

    def batches(self, batch_size, pass_index=0, *, start_batch=0, drop_last=False, device="cpu"):
        import torch

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, pass_index]))
        order = rng.permutation(len(self.records))[self.rank :: self.world_size]
        for b in range(start_batch, self.n_batches(batch_size, pass_index, drop_last=drop_last)):
            chunk = self.records[order[b * batch_size : (b + 1) * batch_size]]
            yield torch.as_tensor(chunk, device=device)

    def load(self):
        return self.records


def _records(batch) -> np.ndarray:
    """Stream records [B*N, D+P+1] float32 of a batch: theta, phi, label bits."""
    b, n = batch.labels.shape
    theta = np.repeat(batch.theta[:, None, :], n, axis=1)  # [B, N, D]
    out = np.zeros((b * n, DIM_THETA + DIM_PHI + 1), dtype=np.float32)
    out[:, :-1] = np.concatenate([theta, batch.phi], axis=-1).reshape(b * n, -1)
    out[:, -1] = batch.labels.reshape(-1).astype(np.int32).view(np.float32)
    return out


def write_data(root: Path, seed: int = 0) -> None:
    """Seeded synthetic design-point data (NpzBatches layout) plus an event store."""
    source = for_scenario(SCENARIO, seed=seed)
    batches = {}
    for k, (partition, fidelity, n_points, n_events) in enumerate(SETS):
        batch = source.generate(n_trials=n_points, n_events=n_events, seed=seed * 100 + k)
        batches[partition, fidelity] = batch
        folder = root / "data" / "batches" / partition
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    store = root / "store"
    store.mkdir()
    np.save(store / "train.npy", _records(batches["train", "lf"]))
    np.save(store / "validation.npy", _records(batches["validation", "lf"]))


def write_configs(root: Path) -> dict[str, Path]:
    """Run configs (few steps, one GP restart); paths relative to each YAML."""
    model = {"kind": "mlp", "architecture": {"hidden_dims": [16, 16]}, "last_layer": {"draws": 200}}
    inference = {f"{p}/{f}": f"runs/train/inference/best/{p}_{f}.npz" for p, f in PREDICTED}
    configs = {
        "train": {
            "model": model,
            "training": {
                "backend": "neural",
                "n_steps": 20,
                "eval_every": 10,
                "batch_size": 4,
                "n_events": 20,
            },
            "data_directory": "data",
            "data_provider": "scripts.smoke:SmokeNpz",
            "output_directory": "runs/train",
            "validation_context_events": 4,
            "selection": "bernoulli_log_loss",
        },
        "mfgp": {
            "levels": [
                {
                    "source": "model_mean",
                    "partition": "gp_train",
                    "fidelity": "lf",
                    "inference": inference["gp_train/lf"],
                },
                {"source": "observed_fraction", "partition": "gp_train", "fidelity": "hf"},
            ],
            "evaluation": [{"partition": "gp_validation", "fidelity": "hf"}],
            "n_restarts": 1,
            "data_directory": "data",
            "output_directory": "runs/mfgp",
        },
        "binomial": {
            "observations": [
                {"partition": "gp_train", "fidelity": "lf"},
                {"partition": "gp_train", "fidelity": "hf"},
            ],
            "evaluation": [{"partition": "gp_validation", "fidelity": "hf"}],
            "covariate": {"inference": inference},
            "n_restarts": 1,
            "max_iters": 30,
            "prediction_draws": 2048,
            "data_directory": "data",
            "output_directory": "runs/binomial",
        },
        "stream": {
            "model": model,
            "training": {
                "batch_size": 64,
                "n_passes": 1,
                "eval_every": 4,
                "device": "cpu",
                "validation_events": 200,
            },
            "event_store": "store",
            "event_source": "scripts.smoke:SmokeEventSource",
            "data_provider": "scripts.smoke:SmokeNpz",
            "data_directory": "data",
            "evaluation": [{"split": "validation", "partition": "gp_validation", "fidelity": "hf"}],
            "history_voxels": 4,
            "output_directory": "runs/stream",
            "dim_theta": DIM_THETA,
            "dim_phi": DIM_PHI,
        },
    }
    configs["within_point"] = {
        "run": "runs/train",
        "within_point_null_draws": 100,
    }
    # Residual learnability test only where the optional lgbm extra is installed.
    if importlib.util.find_spec("lightgbm") is not None:
        residual = {"min_positives": 1, "min_data_in_leaf": 5, "max_trees": 20}
        configs["train"]["residual_test"] = configs["stream"]["residual_test"] = residual
        for run in ("train", "stream"):
            configs[f"residual {run}"] = {
                "run": f"runs/{run}",
                "residual_test": residual,
                "output_directory": f"runs/residual_{run}",
            }
    paths = {}
    for name, config in configs.items():
        paths[name] = root / f"{name.replace(' ', '_')}.yaml"
        paths[name].write_text(yaml.safe_dump(config, sort_keys=False))
    return paths


def _check_json(run: Path, names=("metrics.json", "compute.json")) -> None:
    for name in names:
        path = run / name
        if not path.is_file():
            raise RuntimeError(f"{path} missing")
        if not json.loads(path.read_text()):
            raise RuntimeError(f"{path} is empty")


def stages(root: Path, configs: dict[str, Path]):
    """Groups of independent stages, in order; each stage is (name, CLI arguments, check)."""
    runs = root / "runs"
    checkpoint = runs / "train" / "checkpoints" / "best"
    yield [
        ("train", ["train", configs["train"]], lambda: _check_json(runs / "train")),
        ("stream", ["stream", configs["stream"]], lambda: _check_json(runs / "stream")),
    ]
    predicts = []
    for k, (partition, fidelity) in enumerate(PREDICTED):
        output = runs / "train" / "inference" / "best" / f"{partition}_{fidelity}.npz"
        # The first prediction also writes per-event logits (off by default).
        events = output.parent / f"{partition}_{fidelity}_events" / "index.json" if k == 0 else None

        def exists(output=output, events=events):
            for path in (output, events):
                if path is not None and not path.is_file():
                    raise RuntimeError(f"{path} missing")

        args = ["predict", checkpoint, root / "data", partition, "--fidelity", fidelity]
        args += ["--event-logits"] if k == 0 else []
        predicts.append((f"predict {partition}/{fidelity}", args, exists))
    output = runs / "train"
    check = lambda output=output: _check_json(output, ("within_point.json",))  # noqa: E731
    predicts.append(("within_point", ["within_point", configs["within_point"]], check))
    for run in ("train", "stream"):
        if f"residual {run}" in configs:
            output = runs / f"residual_{run}"
            check = lambda output=output: _check_json(output, ("residual_test.json",))  # noqa: E731
            predicts.append((f"residual {run}", ["residual", configs[f"residual {run}"]], check))
    yield predicts
    yield [
        ("mfgp", ["mfgp", configs["mfgp"]], lambda: _check_json(runs / "mfgp")),
        ("binomial", ["binomial", configs["binomial"]], lambda: _check_json(runs / "binomial")),
    ]


def _run_stage(name, args, check, env) -> float:
    start = time.perf_counter()
    command = [sys.executable, "-m", "core.surrogates", *map(str, args)]
    result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"stage {name} failed (exit {result.returncode}): {' '.join(command)}\n"
            f"{result.stdout[-3000:]}\n{result.stderr[-6000:]}"
        )
    check()
    return time.perf_counter() - start


def run(root: Path) -> list[tuple[str, float]]:
    """Run every stage in ``root`` (must be empty or new); returns (stage, seconds).

    The stages of a group run concurrently (at most 3 single-thread processes).
    """
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty; the smoke check needs a fresh directory")
    start = time.perf_counter()
    write_data(root)
    configs = write_configs(root)
    timings = [("data + configs", time.perf_counter() - start)]
    env = {**os.environ, **THREADS}
    for group in stages(root, configs):
        with ThreadPoolExecutor(len(group)) as pool:
            futures = [(name, pool.submit(_run_stage, name, a, c, env)) for name, a, c in group]
            timings += [(name, future.result()) for name, future in futures]
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    parser.add_argument("--workdir", type=Path, help="run here instead (new or empty, kept)")
    args = parser.parse_args()
    if args.workdir is not None:
        root, cleanup = args.workdir.resolve(), None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="resum_smoke_")
        root = Path(cleanup.name)
    start = time.perf_counter()
    try:
        timings = run(root)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        print(f"(work directory: {root})", file=sys.stderr)
        if cleanup is not None:
            cleanup._finalizer.detach()  # keep the failed run for inspection
        sys.exit(1)
    for name, seconds in timings:
        print(f"OK  {name:28s} {seconds:6.1f} s")
    print(f"OK  {'total (wall)':28s} {time.perf_counter() - start:6.1f} s")
    if cleanup is not None and args.keep:
        cleanup._finalizer.detach()
        print(f"kept {root}")
    elif cleanup is not None:
        cleanup.cleanup()


if __name__ == "__main__":
    main()
