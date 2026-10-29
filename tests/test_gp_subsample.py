"""Seeded GP voxel subsets: helper, batch subset and both GP entry points."""

import json

import numpy as np
import pytest

from core.gp_subsample import subsample_voxels
from data.pseudo_generator import for_scenario
from schemas.data_models import StandardBatch


def test_subsample_voxels_is_seeded_sorted_and_stream_independent():
    np.testing.assert_array_equal(subsample_voxels(5, None, 0, 0), np.arange(5))
    np.testing.assert_array_equal(subsample_voxels(5, 9, 0, 0), np.arange(5))
    a = subsample_voxels(100, 10, 3, 0)
    assert len(a) == len(np.unique(a)) == 10 and np.all(np.diff(a) > 0)
    np.testing.assert_array_equal(a, subsample_voxels(100, 10, 3, 0))
    assert not np.array_equal(a, subsample_voxels(100, 10, 3, 1))
    with pytest.raises(ValueError):
        subsample_voxels(0, 1, 0, 0)


def test_take_keeps_events_and_ragged_counts():
    batch = for_scenario("S1", seed=1).generate(n_trials=4, n_events=6)
    labels = batch.labels.copy()
    labels[1, 3:] = 0
    ragged = StandardBatch(
        mode=batch.mode,
        theta=batch.theta,
        phi=batch.phi,
        labels=labels,
        event_counts=np.array([6, 3, 6, 5]),
    )
    sub = ragged.take(np.array([1, 3]))
    np.testing.assert_array_equal(sub.theta, batch.theta[[1, 3]])
    np.testing.assert_array_equal(sub.phi, batch.phi[[1, 3]])
    np.testing.assert_array_equal(sub.events_per_voxel, [3, 5])
    assert ragged.take(np.array([0, 2])).event_counts is None  # Canonical full form.
    with pytest.raises(ValueError):
        ragged.take(np.array([0.5]))


def test_mfgp_stage_fits_on_voxel_subset(tmp_path):
    pytest.importorskip("GPy")
    from core.surrogates.experiment import run_experiment
    from schemas.surrogates import SurrogateRunConfig

    source = for_scenario("S1", seed=1)
    root = tmp_path / "data"
    sizes = {"lf": 9, "hf": 6}
    for split in ("train", "validation"):
        for fid, n in sizes.items():
            batch = source.generate(n_trials=n if split == "train" else 3, n_events=16)
            folder = root / "batches" / split
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(
                folder / f"{fid}.npz",
                mode=batch.mode.value,
                theta=batch.theta,
                phi=batch.phi,
                labels=batch.labels,
            )
    config = SurrogateRunConfig(
        model={"kind": "cnp", "encoder": {"latent_dim": 8, "hidden_dims": [12]}},
        training={
            "backend": "neural",
            "loss": "bernoulli",
            "n_steps": 2,
            "eval_every": 1,
            "batch_size": 2,
            "n_events": 8,
            "n_context_min": 4,
            "n_context_max": 4,
        },
        mfgp={
            "n_restarts": 1,
            "n_context": 4,
            "levels": [
                {"source": "model_mean", "partition": "train", "fidelity": "lf", "max_voxels": 5},
                {"source": "model_mean", "partition": "train", "fidelity": "hf", "max_voxels": 4},
                {
                    "source": "observed_fraction",
                    "partition": "train",
                    "fidelity": "hf",
                    "max_voxels": 4,
                },
            ],
            "evaluation": [{"partition": "validation", "fidelity": "hf"}],
        },
        data_directory=root,
        output_directory=tmp_path / "run",
        validation_context_events=4,
    )
    out = run_experiment(config)
    with (
        np.load(out / "mfgp/training_arrays.npz") as arrays,
        np.load(root / "batches/train/lf.npz") as full,
    ):
        lf, hf = arrays["index_0"], arrays["index_1"]
        np.testing.assert_array_equal(lf, subsample_voxels(9, 5, 0, 0))
        np.testing.assert_array_equal(hf, subsample_voxels(6, 4, 0, 1))
        # Levels on the same partition share one subset.
        np.testing.assert_array_equal(arrays["index_2"], hf)
        np.testing.assert_array_equal(arrays["X_0"], full["theta"][lf])
        assert arrays["X_1"].shape[0] == 4
    levels = json.loads((out / "mfgp/levels.json").read_text())
    assert [r["voxels"] for r in levels] == [5, 4, 4]
    with np.load(out / "mfgp/evaluation_0.npz") as validation:
        assert len(validation["mean"]) == 3  # Validation is never subsampled.


def test_count_gp_fits_on_voxel_subset(tmp_path):
    pytest.importorskip("GPy")
    from core.binomial_gp import BinomialGP
    from core.count_gp_experiment import run_count_gp
    from data.optical_counts import prepare_optical_counts
    from schemas.count_gp import CountGPConfig
    from tests.test_optical_pipeline import config, fixture_file

    c = config(tmp_path)
    for i in range(8):
        fixture_file(
            c.source.directory,
            "lf",
            i,
            center=(float(i), 0.0, 0.0),
            n=8,
            hit_ids=(1,) if i % 2 else (),
            detectors=(10,) if i % 2 else (),
        )
    for i in range(4):
        fixture_file(c.source.directory, "hf", i, center=(float(i), 1.0, 0.0), n=12)
    c.split.lf_train_only = True
    c.split.hf_train_count = 2
    prepared = prepare_optical_counts(c)
    n_train = len(prepared[0]["train"]["theta"])
    cfg = CountGPConfig(
        data=c,
        output_directory=tmp_path / "run",
        prediction_draws=2048,
        backend={"max_iters": 5, "n_restarts": 1, "max_training_voxels": 4, "seed": 7},
    )
    out = run_count_gp(cfg, prepared)
    subset = np.load(out / "training_voxel_subset.npy")
    np.testing.assert_array_equal(subset, subsample_voxels(n_train, 4, 7, 0))
    gp = BinomialGP.load(out / "model.pkl")
    np.testing.assert_array_equal(gp.training_trials, prepared[0]["train"]["trials"][subset])
    with np.load(out / "train_counts.npz") as saved:
        assert len(saved["theta"]) == n_train  # Snapshot keeps the full partition.
