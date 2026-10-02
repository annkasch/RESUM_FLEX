"""Training-only normalization and spatial exclusion across simulation budgets."""

import json

import numpy as np
import pytest

from data.optical_pipeline import load_prepared_batch
from data.optical_transfer import prepare_optical_transfer_data
from schemas.optical_transfer import OpticalTransferDataConfig
from tests.test_optical_pipeline import fixture_file


def test_transfer_keeps_zeros_and_excludes_nearby_voxels(tmp_path):
    source = tmp_path / "development"
    external = tmp_path / "external"
    for i in range(10):
        fixture_file(
            source,
            run=i,
            center=(i * 0.1, 0, 0),
            n=8,
            hit_ids=() if i % 2 == 0 else (1,),
            detectors=() if i % 2 == 0 else (10,),
        )
    # First point is less than 5 mm from a development voxel.
    fixture_file(external, run=1, center=(0.003, 0, 0), n=12)
    fixture_file(external, run=2, center=(2, 0, 0), n=12, hit_ids=(), detectors=())
    fixture_file(external, run=3, center=(3, 0, 0), n=12)
    cfg = OpticalTransferDataConfig(
        source={"directories": {"lf": [source / "lf"]}},
        test_source={"directories": {"lf": [external / "lf"]}},
        output_directory=tmp_path / "prepared",
        training_primaries=8,
        test_primaries=12,
        split={
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
            "test_fraction": 0,
            "manifest": tmp_path / "manifest.json",
        },
    )
    prepared = prepare_optical_transfer_data(cfg)
    train, val, test = [prepared.batches[s]["lf"] for s in ("train", "validation", "test")]
    assert (train.batch_size, val.batch_size, test.batch_size) == (8, 2, 2)
    assert test.n_events == 12 and (test.labels.sum(1) == 0).any()
    fit_files = prepared.normalization["fit_files"]
    assert len(fit_files) == 8
    theta = prepared.normalization["theta"]
    # Normalized training mean is zero; neither external test positions affect it.
    np.testing.assert_allclose(train.theta.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(
        test.theta[:, 0], (np.array([2, 3]) - theta["offset"][0]) / theta["scale"][0]
    )
    loaded = load_prepared_batch(cfg.output_directory / "batches/test/lf.npz")
    np.testing.assert_array_equal(test.labels, loaded.labels)
    manifest = json.loads((cfg.output_directory / "test_source_manifest.json").read_text())
    assert len(manifest["excluded"]) == 1
    assert "spatial holdout" in manifest["excluded"][0]["reason"]
    # Repreparation retains the split and produces identical normalization.
    repeated = prepare_optical_transfer_data(cfg)
    assert repeated.normalization == prepared.normalization


def test_transfer_requires_exact_simulation_budgets(tmp_path):
    source, external = tmp_path / "dev", tmp_path / "test"
    fixture_file(source, n=8)
    fixture_file(external, n=12, center=(10, 0, 0))
    cfg = OpticalTransferDataConfig(
        source={"directories": {"lf": [source / "lf"]}},
        test_source={"directories": {"lf": [external / "lf"]}},
        split={"train_fraction": 0.8, "validation_fraction": 0.2, "test_fraction": 0},
    )
    with pytest.raises(ValueError, match="1500 single-primary events"):
        prepare_optical_transfer_data(cfg)


def test_transfer_adds_hf_without_changing_lf_normalization(tmp_path):
    source, external = tmp_path / "dev", tmp_path / "test"
    for i in range(10):
        fixture_file(source, run=i, center=(i * 0.1, 0, 0), n=8)
    fixture_file(external, run=1, center=(2.003, 0, 0), n=12)
    fixture_file(external, run=2, center=(9, 0, 0), n=12)
    cfg = OpticalTransferDataConfig(
        source={"directories": {"lf": [source / "lf"]}},
        test_source={"directories": {"lf": [external / "lf"]}},
        output_directory=tmp_path / "prepared",
        training_primaries=8, hf_primaries=20, test_primaries=12,
        normalization={"fit_fidelities": ["lf"]},
        split={"train_fraction": .8, "validation_fraction": .2, "test_fraction": 0,
               "manifest": tmp_path / "manifest.json"},
    )
    before = prepare_optical_transfer_data(cfg)
    for i in range(4):
        fixture_file(source, fidelity="hf", run=i, center=(2+i, 0, 0), n=20,
                     hit_ids=(), detectors=())
    cfg.source.directories["hf"] = [source / "hf"]
    cfg.split.hf_train_count = 2
    after = prepare_optical_transfer_data(cfg)
    assert after.normalization == before.normalization
    assert after.batches["train"]["hf"].batch_size == 2
    assert after.batches["validation"]["hf"].batch_size == 2
    assert after.batches["train"]["hf"].labels.sum() == 0
    assert after.batches["test"]["lf"].batch_size == 1
    np.testing.assert_array_equal(before.batches["train"]["lf"].theta,
                                  after.batches["train"]["lf"].theta)
    cfg.hf_primaries = 21
    with pytest.raises(ValueError, match="21 single-primary events"):
        prepare_optical_transfer_data(cfg)
