"""Optical fixtures verify event alignment, leakage prevention and integration."""

import json
from dataclasses import replace

import numpy as np
import pytest

from data.optical_features import AffineTransform, cylindrical, make_features, make_labels
from data.optical_pipeline import load_prepared_batch, prepare_optical_data
from data.optical_reader import read_optical_runs, read_run
from data.optical_split import assign_splits, integer_targets, save_manifest
from schemas.config import load_config
from schemas.optical import ANOMALOUS_FILE, OpticalDataConfig

h5py = pytest.importorskip("h5py")


def fixture_file(
    root,
    fidelity="lf",
    run=1,
    center=(0.1, 0.2, 0.3),
    n=8,
    hit_ids=(3, 3, 1),
    detectors=(10, 11, 10),
):
    x, y, z = center
    path = root / fidelity / f"run{run:03}_x{x:+.3f}_y{y:+.3f}_z{z:+.3f}.stp.lh5"
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.arange(n, dtype=np.int32)[::-1]
    with h5py.File(path, "w") as h:
        h["number_of_simulated_events"] = n

        def ds(name, values, units=None):
            d = h.create_dataset(name, data=values)
            if units:
                d.attrs["units"] = units

        ds("vtx/evtid", ids)
        for i, axis in enumerate("xyz"):
            ds(f"vtx/{axis}loc", np.full(n, center[i]) + np.linspace(-0.002, 0.002, n), "m")
            ds(f"vtx/p{axis}", np.arange(n) * 1e-6 + (i + 1) * 1e-5, "MeV")
        ds("vtx/time", np.zeros(n), "ns")
        ds("vtx/n_part", np.ones(n, dtype=np.int32))
        ds("stp/optical/evtid", np.array(hit_ids, dtype=np.int32))
        ds("stp/optical/det_uid", np.array(detectors, dtype=np.int32))
        ds("stp/optical/time", np.arange(len(hit_ids), dtype=float), "ns")
        ds("stp/optical/wavelength", np.full(len(hit_ids), 500.0), "nm")
    return path


def config(tmp_path, **updates):
    return OpticalDataConfig.model_validate(
        {
            "source": {"directory": str(tmp_path / "source")},
            "output_directory": str(tmp_path / "output"),
            "split": {"manifest": str(tmp_path / "manifest.json")},
            **updates,
        }
    )


def dataset(tmp_path, number=10):
    c = config(tmp_path)
    for i in range(number):
        for fidelity in ("lf", "hf"):
            fixture_file(
                c.source.directory,
                fidelity,
                i,
                (0.05 * i, 0.01 * i, 0.2),
                n=8 if fidelity == "lf" else 12,
            )
    return c


def test_reader_preserves_event_pairs_and_collapses_hits(tmp_path):
    c = config(tmp_path)
    p = fixture_file(c.source.directory)
    r = read_run(p, c.source.directory)
    assert r.hit_event_ids.tolist() == [3, 3, 1]
    assert make_labels(r, c).tolist() == [0, 0, 0, 0, 1, 0, 1, 0]
    c.target.kind = "single_channel"
    c.target.detector_id = 11
    assert make_labels(r, c).sum() == 1
    assert make_labels(r, c)[4] == 1
    zero = read_run(
        fixture_file(c.source.directory, run=2, hit_ids=(), detectors=()), c.source.directory
    )
    assert make_labels(zero, c).sum() == 0


@pytest.mark.parametrize(
    "corruption", ["missing", "units", "length", "nan", "duplicate", "unknown"]
)
def test_reader_rejects_invalid_input_with_filename(tmp_path, corruption):
    c = config(tmp_path)
    p = fixture_file(c.source.directory)
    with h5py.File(p, "a") as h:
        if corruption == "missing":
            del h["vtx/px"]
        elif corruption == "units":
            h["vtx/xloc"].attrs["units"] = "mm"
        elif corruption == "length":
            del h["vtx/time"]
            d = h.create_dataset("vtx/time", data=[0.0])
            d.attrs["units"] = "ns"
        elif corruption == "nan":
            h["vtx/xloc"][0] = np.nan
        elif corruption == "duplicate":
            h["vtx/evtid"][0] = 0
        elif corruption == "unknown":
            h["stp/optical/evtid"][0] = 99
    with pytest.raises(ValueError) as error:
        read_run(p, c.source.directory)
    assert p.name in str(error.value)


def test_anomaly_always_excluded(tmp_path):
    c = config(tmp_path)
    fixture_file(c.source.directory)
    path = c.source.directory / ANOMALOUS_FILE
    path.write_text("Excluded before opening, even if corrupt")
    c.source.exclude_files = []
    runs, excluded = read_optical_runs(c.source)
    assert len(runs) == 1 and excluded[0]["file"] == ANOMALOUS_FILE


def test_split_shared_voxels_reuse_and_explicit_update(tmp_path):
    c = dataset(tmp_path)
    runs, _ = read_optical_runs(c.source)
    # Simulated range midpoints differ; identity still uses nominal centers.
    runs[0] = replace(runs[0], measured_center=runs[0].measured_center + 0.001)
    assignments, payload = assign_splits(runs, c)
    assert assignments == assign_splits(runs, c)[0]
    assert payload["actual_counts"] == [[7, 7], [2, 2], [1, 1]]
    for a in runs:
        for b in runs:
            if np.array_equal(a.nominal_center, b.nominal_center):
                assert assignments[a.file] == assignments[b.file]
    save_manifest(c.split.manifest, payload)
    assert assignments == assign_splits(runs, c)[0]
    changed = [replace(runs[0], fingerprint="changed"), *runs[1:]]
    with pytest.raises(ValueError, match="Source files changed"):
        assign_splits(changed, c)
    assert assignments == assign_splits(changed, c, update=True)[0]
    extra_path = fixture_file(c.source.directory, "lf", 99, (9.0, 0.0, 0.0))
    extra = read_run(extra_path, c.source.directory)
    with pytest.raises(ValueError, match="Source files changed"):
        assign_splits([*runs, extra], c)
    updated, _ = assign_splits([*runs, extra], c, update=True)
    assert all(updated[k] == v for k, v in assignments.items())


def test_azimuth_free_groups_and_rejects_leaky_manifest(tmp_path):
    c = config(tmp_path)
    a = read_run(fixture_file(c.source.directory, center=(1.0, 0.0, 0.2)), c.source.directory)
    b = read_run(
        fixture_file(c.source.directory, run=2, center=(0.0, 1.0, 0.2)), c.source.directory
    )
    _, payload = assign_splits([a, b], c)
    payload["files"][0]["split"] = "train"
    payload["files"][1]["split"] = "test"
    save_manifest(c.split.manifest, payload)
    c.theta.coordinates = "cylindrical"
    c.theta.cylindrical.include_azimuth = False
    with pytest.raises(ValueError, match="Equivalent voxel"):
        assign_splits([a, b], c)
    c.split.manifest = tmp_path / "new_manifest.json"
    assignments, _ = assign_splits([a, b], c)
    assert assignments[a.file] == assignments[b.file]
    near = replace(b, nominal_center=np.array([0, 1 + 5e-7, 0.2]))
    assert len(set(assign_splits([a, near], c)[0].values())) == 1


def test_features_offsets_angles_and_momentum(tmp_path):
    c = config(tmp_path)
    r = read_run(fixture_file(c.source.directory, center=(0, 1, 0.2)), c.source.directory)
    theta, phi, info = make_features(r, c)
    np.testing.assert_allclose(phi[:, :3], r.positions - theta)
    assert phi[0, 0] < 0 < phi[-1, 0]
    assert info["phi"]["dimension"] == 6
    c.phi.vertex.coordinates = "cylindrical"
    _, local, _ = make_features(r, c)
    delta = r.positions - r.nominal_center
    np.testing.assert_allclose(local[:, 0], delta[:, 1], atol=1e-12)
    np.testing.assert_allclose(local[:, 1], -delta[:, 0], atol=1e-12)
    a, _, _ = cylindrical(np.array([[-1, 1e-10, 0], [-1, -1e-10, 0], [0, 0, 0]]))
    np.testing.assert_allclose(a[0], a[1], atol=1e-9)
    np.testing.assert_array_equal(a[2], [0, 0, 1, 0])
    c.theta.coordinates = "cylindrical"
    assert len(make_features(r, c)[0]) == 4
    c.theta.cylindrical.angle_encoding = "radians"
    assert len(make_features(r, c)[0]) == 3
    c.phi.vertex.representation = "absolute"
    _, absolute, _ = make_features(r, c)
    np.testing.assert_allclose(absolute[:, 0], np.hypot(*r.positions[:, :2].T))
    c.phi.momentum.representation = "direction_and_magnitude"
    _, phi, _ = make_features(r, c)
    np.testing.assert_allclose(np.linalg.norm(phi[:, -4:-1], axis=1), 1)
    with pytest.raises(ValueError, match="zero momentum"):
        make_features(replace(r, momenta=np.zeros_like(r.momenta)), c)


@pytest.mark.parametrize("method", ["standard", "minmax", "none"])
def test_normalization_constant_and_serialization(method):
    train = np.array([[1.0, 5.0], [3.0, 5.0]])
    t = AffineTransform.fit(train, method)
    saved = AffineTransform(**json.loads(json.dumps(t.to_dict())))
    np.testing.assert_array_equal(t.transform(train), saved.transform(train))
    assert t.scale[1] == 1
    assert saved.transform(np.array([[100.0, 5.0]]))[0, 0] > 1


def test_prepare_roundtrip_training_only_and_reports(tmp_path):
    c = dataset(tmp_path)
    result = prepare_optical_data(c)
    assert sum(b.batch_size for fs in result.batches.values() for b in fs.values()) == 20
    train_files = {name for meta in result.metadata["train"].values() for name in meta["files"]}
    assert train_files == set(result.normalization["fit_files"])
    runs, _ = read_optical_runs(c.source)
    expected = np.array([r.nominal_center for r in runs if r.file in train_files]).mean(0)
    np.testing.assert_allclose(result.normalization["theta"]["offset"], expected)
    expected_phi = np.concatenate(
        [make_features(r, c)[1] for r in runs if r.file in train_files]
    ).mean(0)
    np.testing.assert_allclose(result.normalization["phi"]["offset"], expected_phi)
    loaded = load_prepared_batch(c.output_directory / "batches/train/lf.npz")
    np.testing.assert_array_equal(loaded.phi, result.batches["train"]["lf"].phi)
    source = result.training_source()
    probe = source.generate(2, 8, seed=1)
    assert all(len(np.unique(trial, axis=0)) == 8 for trial in probe.phi)
    assert (c.output_directory / "reports/test/lf_channels.csv").exists()
    assert (c.output_directory / "reports/development/train/lf_coverage.png").exists()
    before = c.split.manifest.read_text()
    c.phi.vertex.representation = "absolute"
    prepare_optical_data(c)
    assert before == c.split.manifest.read_text()


def test_design_only_and_unequal_event_counts(tmp_path):
    c = dataset(tmp_path)
    c.phi.vertex.representation = c.phi.momentum.representation = "omit"
    result = prepare_optical_data(c)
    assert result.batches["train"]["lf"].phi is None
    assert result.batches["train"]["lf"].mode.value == "design_only"
    # A repeated run must share the voxel split, exposing ragged input.
    fixture_file(c.source.directory, "lf", 98, (0.0, 0.0, 0.2), n=9)
    with pytest.raises(ValueError, match="unequal event counts"):
        prepare_optical_data(c, update_manifest=True)


def test_configs_and_integer_targets(tmp_path, legacy_config_path):
    assert load_config(legacy_config_path).data is None
    assert integer_targets(85, (0.7, 0.15, 0.15)).tolist() == [59, 13, 13]
    assert integer_targets(22, (0.7, 0.15, 0.15)).tolist() == [16, 3, 3]
    with pytest.raises(ValueError, match="detector_id"):
        config(tmp_path, target={"kind": "single_channel"})
    with pytest.raises(ValueError, match="sum to 1"):
        config(tmp_path, split={"train_fraction": 0.8})


def test_cnp_context_target_and_training_smoke(tmp_path):
    import torch

    from core.surrogate_cnp import build_cnp, split_context_target
    from core.training import train_cnp
    from data.batch_source import FixedBatchSource
    from schemas.config import CNPConfig, EncoderConfig, TrainingConfig
    from schemas.data_models import InputMode, StandardBatch

    torch.set_num_threads(1)
    c = config(tmp_path)
    r = read_run(fixture_file(c.source.directory), c.source.directory)
    theta, phi, _ = make_features(r, c)
    transform = AffineTransform.fit(phi, "standard")
    batch = StandardBatch(
        mode=InputMode.FULL,
        theta=theta[None, :],
        phi=transform.transform(phi)[None, :, :],
        labels=make_labels(r, c)[None, :],
    )
    ctx, tgt = split_context_target(batch, n_context=3, seed=3)
    assert set(map(tuple, ctx.phi[0])).isdisjoint(set(map(tuple, tgt.phi[0])))
    model = build_cnp(EncoderConfig(latent_dim=8, hidden_dims=[16]), dim_theta=3, dim_phi=6)
    hist = train_cnp(
        model,
        FixedBatchSource(batch),
        cnp_config=CNPConfig(n_context_min=2, n_context_max=4),
        training_config=TrainingConfig(n_steps=2, batch_size=1, n_events_per_trial=8, eval_every=0),
    )
    assert np.isfinite(hist["loss"]).all()


def test_multiple_input_folders_deduplicate_and_keep_identities(tmp_path):
    from schemas.optical import SourceConfig

    first = fixture_file(tmp_path / "a", "lf", run=1)
    second = fixture_file(tmp_path / "b", "lf", run=2, center=(.4, .5, .6))
    high = fixture_file(tmp_path / "c", "hf", run=3)
    alias = tmp_path / "alias"
    alias.mkdir()
    (alias / first.name).symlink_to(first)
    source = SourceConfig(directories={"lf": [first.parent, second.parent, alias], "hf": [high.parent]})
    runs, excluded = read_optical_runs(source)
    assert len(runs) == 3 and not excluded
    assert {r.file for r in runs} == {f"lf/{first.name}", f"lf/{second.name}", f"hf/{high.name}"}
    source.exclude_files = [f"lf/{second.name}"]
    runs, excluded = read_optical_runs(source)
    assert len(runs) == 2 and len(excluded) == 1


def test_folder_lists_reject_ambiguous_sources(tmp_path):
    from schemas.optical import SourceConfig
    import shutil

    with pytest.raises(ValueError, match="either directory"):
        SourceConfig(directory=tmp_path, directories={"lf": [tmp_path]})
    with pytest.raises(ValueError, match="either directory"):
        SourceConfig()
    with pytest.raises(ValueError, match="at least one folder"):
        SourceConfig(directories={"lf": []})
    first = fixture_file(tmp_path / "a")
    other = tmp_path / "other"
    other.mkdir()
    shutil.copyfile(first, other / first.name)
    with pytest.raises(ValueError, match="Ambiguous"):
        read_optical_runs(SourceConfig(directories={"lf": [first.parent, other]}))
    with pytest.raises(ValueError, match="LF and HF"):
        read_optical_runs(SourceConfig(directories={"lf": [first.parent], "hf": [first.parent]}))
    with pytest.raises(FileNotFoundError):
        read_optical_runs(SourceConfig(directories={"lf": [tmp_path / "missing"]}))


def test_preparation_config_resolves_folder_lists(tmp_path):
    from schemas.optical import load_optical_config

    path = tmp_path / "data.yaml"
    path.write_text('source:\n  directories:\n    lf: [lf1, lf2]\n    hf: [hf]\noutput_directory: prepared\nsplit:\n  manifest: split.json\n')
    c = load_optical_config(path)
    assert c.source.directories['lf'] == [tmp_path / 'lf1', tmp_path / 'lf2']
    assert c.output_directory == tmp_path / 'prepared'
    assert c.split.manifest == tmp_path / 'split.json'


def test_automatic_preparation_recovers_batches_and_keeps_split(tmp_path):
    from data.optical_pipeline import ensure_prepared_optical_data

    c = config(tmp_path)
    for i in range(4):
        fixture_file(c.source.directory, 'lf', i, center=(float(i), 0., 0.))
    for i in range(5):
        fixture_file(c.source.directory, 'hf', i, center=(float(i), 1., 0.))
    c.split.lf_train_only = True
    c.split.hf_train_count = 2
    out = ensure_prepared_optical_data(c)
    manifest = json.loads(c.split.manifest.read_text())
    assert all(r['split'] == 'train' for r in manifest['files'] if r['fidelity'] == 'lf')
    assert sum(r['split'] == 'train' and r['fidelity'] == 'hf' for r in manifest['files']) == 2
    assert not (out / 'batches/validation/lf.npz').exists()
    path = out / 'batches/train/lf.npz'
    before = path.stat().st_mtime_ns
    ensure_prepared_optical_data(c)
    assert path.stat().st_mtime_ns == before
    path.unlink()
    ensure_prepared_optical_data(c)
    assert path.exists()
    assert json.loads(c.split.manifest.read_text()) == manifest
    # Rebuilding after losing the manifest also reproduces the explicit policy.
    c.split.manifest.unlink()
    ensure_prepared_optical_data(c)
    assert json.loads(c.split.manifest.read_text()) == manifest
