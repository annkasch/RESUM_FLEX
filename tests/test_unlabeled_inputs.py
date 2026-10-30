"""Files without optical hit data: cache, gp_unlabeled role, inputs-only loading, inference."""

import json

import numpy as np
import pytest

from data.event_cache import INDEX_DTYPE, EventCache
from data.optical_pipeline import load_partition, prepare_optical_data
from data.optical_reader import read_run
from data.optical_view import OpticalView
from tests.test_optical_pipeline import fixture_file, role_config, role_folder

h5py = pytest.importorskip("h5py")


def strip_hits(folder):
    """Remove the optical hit group, as in inputs-only simulations (LF_primaries)."""
    for path in sorted(folder.glob("*.stp.lh5")):
        with h5py.File(path, "r+") as h:
            del h["stp"]
    return folder


def prepared(tmp_path, unlabeled_role="gp_unlabeled"):
    labeled = role_folder(tmp_path / "labeled", 6)
    unlabeled = strip_hits(role_folder(tmp_path / "inputs", 4, offset=1.0, first_run=10))
    c = role_config(
        tmp_path,
        [
            {"path": str(labeled), "fractions": {"cnp_train": 0.5, "cnp_validation": 0.5}},
            {"path": str(unlabeled), "fractions": {unlabeled_role: 1.0}},
        ],
    )
    return c, prepare_optical_data(c)


def test_cache_marks_files_without_hits_as_unlabeled(tmp_path):
    path = fixture_file(tmp_path / "source", run=1)
    strip_hits(path.parent)
    with pytest.raises(ValueError, match="no optical hit data"):
        read_run(path, tmp_path / "source")  # the labeled reader still requires hits
    cache = EventCache(tmp_path / "cache")
    assert cache.update([path], workers=1) == 1
    row = cache.locate([path])[0]
    assert not cache.index["labeled"][row] and cache.events(row).shape == (8, 6)
    with pytest.raises(ValueError, match="labels are undefined"):
        cache.labels(row)
    with pytest.raises(ValueError, match="no optical hit data"):
        cache.hits(row)
    labeled = fixture_file(tmp_path / "source", run=2, center=(0.5, 0.2, 0.3))
    cache.update([path, labeled], workers=1)
    assert cache.compact() == 1  # raw blocks are copied without label checks
    rows = cache.locate([path, labeled])
    assert cache.index["labeled"][rows].tolist() == [False, True]
    assert cache.labels(rows[1]).sum() == 2


def test_version_one_cache_is_upgraded_as_labeled(tmp_path):
    path = fixture_file(tmp_path / "source", run=1)
    cache = EventCache(tmp_path / "cache")
    cache.update([path], workers=1)
    names = [n for n in INDEX_DTYPE.names if n != "labeled"]
    old = np.zeros(len(cache.index), dtype=[(n, INDEX_DTYPE[n]) for n in names])
    for n in names:
        old[n] = cache.index[n]
    np.save(tmp_path / "cache/index.npy", old)
    meta = json.loads((tmp_path / "cache/cache.json").read_text())
    (tmp_path / "cache/cache.json").write_text(json.dumps({**meta, "version": 1}))
    upgraded = EventCache(tmp_path / "cache")
    assert upgraded.index["labeled"].all()
    assert upgraded.update([path], workers=1) == 0  # nothing is read again


def test_gp_unlabeled_role_loads_inputs_only(tmp_path):
    c, result = prepared(tmp_path)
    out = c.output_directory
    assert result.normalization["fit_files"] == sorted(
        r["file"]
        for r in json.loads((out / "manifest_snapshot.json").read_text())["files"]
        if r["split"] == "cnp_train"
    )
    with pytest.raises(ValueError, match="labels are undefined"):
        load_partition(out, "gp_unlabeled", "lf")
    batch, paths = OpticalView(out).load_inputs("gp_unlabeled", "lf")
    assert batch.batch_size == 4 and len(paths) == 4 and not batch.labels.any()
    labeled, _ = OpticalView(out).load_inputs("cnp_train", "lf")
    np.testing.assert_array_equal(labeled.phi, load_partition(out, "cnp_train", "lf")[0].phi)
    assert not (out / "reports/development/gp_unlabeled").exists()


def test_unlabeled_files_rejected_in_labeled_roles(tmp_path):
    with pytest.raises(ValueError, match="without optical hit data are in labeled partitions"):
        prepared(tmp_path, unlabeled_role="gp_train")


def _checkpoint(tmp_path, kind, normalization):
    from core.surrogates import build_surrogate
    from core.surrogates.checkpoints import save_surrogate

    spec = (
        {"kind": "mlp", "architecture": {"hidden_dims": [8]}}
        if kind == "mlp"
        else {"kind": "cnp", "encoder": {"type": "mlp", "latent_dim": 4, "hidden_dims": [8]}}
    )
    model = build_surrogate(spec, 3, 6, seed=0)
    model.metadata = {
        "normalization.json": normalization,
        "step": 7,
        "data_provider": "data.optical_provider:OpticalDataset",
    }
    path = tmp_path / f"run_{kind}" / "checkpoints" / "best"
    save_surrogate(path, model)
    return path, model


def test_inference_writes_mean_probability_per_voxel(tmp_path):
    from core.surrogates.inference import predict_partition

    c, result = prepared(tmp_path)
    checkpoint, model = _checkpoint(tmp_path, "mlp", result.normalization)
    path = predict_partition(checkpoint, c.output_directory, "gp_unlabeled")
    assert path == tmp_path / "run_mlp/inference/best/gp_unlabeled_lf.npz"
    batch, _ = OpticalView(c.output_directory).load_inputs("gp_unlabeled", "lf")
    with np.load(path) as arrays:
        np.testing.assert_allclose(arrays["mean_probability"], model.predict(batch).mean)
        assert arrays["coordinates"].shape == (4, 3)
        assert (arrays["event_counts"] == 8).all()
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["step"] == 7 and meta["voxels"] == 4
    with pytest.raises(FileExistsError):
        predict_partition(checkpoint, c.output_directory, "gp_unlabeled")


def test_inference_rejects_context_models_and_other_normalization(tmp_path):
    from core.surrogates.inference import predict_partition

    c, result = prepared(tmp_path)
    cnp, _ = _checkpoint(tmp_path, "cnp", result.normalization)
    with pytest.raises(ValueError, match="context-free"):
        predict_partition(cnp, c.output_directory, "gp_unlabeled")
    other = {
        **result.normalization,
        "theta": {**result.normalization["theta"], "offset": [0, 0, 0]},
    }
    mlp, _ = _checkpoint(tmp_path, "mlp", other)
    with pytest.raises(ValueError, match="Normalization"):
        predict_partition(mlp, c.output_directory, "gp_unlabeled")


def hf_layout(tmp_path, *, hf_validation_offset=3.0):
    """LF CNP voxels, unlabeled LF voxels, nested HF voxels and HF validation/test voxels."""
    labeled = role_folder(tmp_path / "labeled", 6)
    unlabeled = strip_hits(role_folder(tmp_path / "inputs", 4, offset=1.0, first_run=10))
    hf = role_folder(tmp_path / "hf", 3, offset=1.0, first_run=20)  # at unlabeled positions
    hf_val = role_folder(tmp_path / "hf_val", 4, offset=hf_validation_offset, first_run=30)
    return role_config(
        tmp_path,
        [
            {"path": str(labeled), "fractions": {"cnp_train": 0.5, "cnp_validation": 0.5}},
            {"path": str(unlabeled), "fractions": {"gp_unlabeled": 1.0}},
            {"path": str(hf), "fidelity": "hf", "fractions": {"gp_train": 1.0}},
            {"path": str(hf_val), "fidelity": "hf",
             "fractions": {"gp_validation": 0.5, "gp_test": 0.5}},
        ],
    )


def test_hf_roles_allow_nested_gp_design(tmp_path):
    c = hf_layout(tmp_path)
    result = prepare_optical_data(c)
    out = c.output_directory
    for role, n in [("gp_train", 3), ("gp_validation", 2), ("gp_test", 2)]:
        batch, _ = load_partition(out, role, "hf")
        assert batch.batch_size == n
    assert all(f.startswith("lf/") for f in result.normalization["fit_files"])
    # The HF training voxels sit at unlabeled LF positions (nested design).
    hf_theta = load_partition(out, "gp_train", "hf")[0].theta
    lf_theta = OpticalView(out).load_theta("gp_unlabeled", "lf")[0]
    assert all(any(np.allclose(t, u) for u in lf_theta) for t in hf_theta)


def test_shared_positions_outside_gp_training_are_rejected(tmp_path):
    # HF validation voxels at the positions of unlabeled GP training voxels: leakage.
    with pytest.raises(ValueError, match="shared by roles"):
        prepare_optical_data(hf_layout(tmp_path, hf_validation_offset=1.0))


def test_two_level_gp_from_inference_and_hf(tmp_path):
    pytest.importorskip("GPy")
    from core.surrogates.inference import predict_partition
    from core.surrogates.mfgp_stage import run_mfgp_config
    from schemas.surrogates import MFGPRunConfig

    c = hf_layout(tmp_path)
    result = prepare_optical_data(c)
    checkpoint, _ = _checkpoint(tmp_path, "mlp", result.normalization)
    inference = predict_partition(checkpoint, c.output_directory, "gp_unlabeled")
    settings = dict(
        levels=[
            {"source": "model_mean", "partition": "gp_unlabeled", "inference": str(inference)},
            {"source": "observed_fraction", "partition": "gp_train", "fidelity": "hf"},
        ],
        n_restarts=1,
        data_directory=c.output_directory,
        data_provider="data.optical_provider:OpticalDataset",
        output_directory=tmp_path / "gp",
    )
    with pytest.raises(ValueError, match="blinded"):
        MFGPRunConfig(**settings, evaluation=[{"partition": "gp_test", "fidelity": "hf"}])
    out = run_mfgp_config(
        MFGPRunConfig(**settings, evaluation=[{"partition": "gp_validation", "fidelity": "hf"}])
    )
    rows = json.loads((out / "metrics.json").read_text())
    assert [(r["partition"], r["fidelity"], r["level"], r["voxels"]) for r in rows] == [
        ("gp_validation", "hf", 1, 2)
    ]
    with np.load(out / "training_arrays.npz") as arrays:
        assert arrays["X_0"].shape[0] == 4 and arrays["X_1"].shape[0] == 3
