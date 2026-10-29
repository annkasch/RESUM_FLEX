"""Unequal event counts per voxel: padded batches with explicit event_counts."""


import numpy as np
import pytest

from core.mixup import ClassAwareMixupSource
from core.surrogate_cnp import split_context_target
from core.surrogates import Episode, build_surrogate, evaluate_surrogate
from core.surrogates.base import EventPrediction
from core.target_sampling import RealTargetSampler
from data.batch_source import FixedBatchSource
from data.pseudo_generator import for_scenario
from schemas.data_models import InputMode, StandardBatch
from schemas.surrogates import NeuralTraining

COUNTS = np.array([30, 18, 24, 12])


def ragged(seed=0, counts=COUNTS):
    """Pseudo data truncated to per-voxel counts, padded with zeros."""
    source = for_scenario("S1", seed=seed)
    full = source.generate(n_trials=len(counts), n_events=int(max(counts)))
    valid = np.arange(full.n_events)[None, :] < counts[:, None]
    labels = np.where(valid, full.labels, 0)
    labels[:, 0] = 1  # every voxel has a positive and a negative
    labels[:, 1] = 0
    phi = np.where(valid[..., None], full.phi, 0.0)
    batch = StandardBatch(
        mode=full.mode, theta=full.theta, phi=phi, labels=labels, event_counts=counts
    )
    return source, batch


def test_schema_validates_counts_and_padding():
    _, b = ragged()
    assert b.is_ragged and b.n_events == 30
    np.testing.assert_array_equal(b.events_per_voxel, COUNTS)
    values = np.arange(b.labels.size, dtype=float).reshape(b.labels.shape)
    expected = [values[i, :n].mean() for i, n in enumerate(COUNTS)]
    np.testing.assert_allclose(b.voxel_mean(values), expected)
    bad = b.labels.copy()
    bad[3, 20] = 1
    with pytest.raises(ValueError, match="Padded events"):
        StandardBatch(mode=b.mode, theta=b.theta, phi=b.phi, labels=bad, event_counts=COUNTS)
    for counts in (np.array([30, 0, 24, 12]), np.array([31, 18, 24, 12]), COUNTS[:3]):
        with pytest.raises(ValueError, match="event_counts"):
            StandardBatch(
                mode=b.mode, theta=b.theta, phi=b.phi, labels=b.labels, event_counts=counts
            )
    full = StandardBatch(
        mode=InputMode.DESIGN_ONLY,
        theta=np.zeros((2, 1)),
        labels=np.zeros((2, 5), dtype=int),
        event_counts=np.array([5, 5]),
    )
    assert full.event_counts is None and not full.is_ragged


def test_split_uses_each_voxels_own_events():
    _, b = ragged()
    ctx, tgt = split_context_target(b, 6, seed=3)
    assert ctx.n_events == 6 and not ctx.is_ragged
    np.testing.assert_array_equal(tgt.events_per_voxel, COUNTS - 6)
    for i, n in enumerate(COUNTS):
        # Context and target partition exactly the valid events of the voxel.
        rows = np.concatenate([ctx.phi[i], tgt.phi[i, : n - 6]])
        np.testing.assert_array_equal(np.sort(rows[:, 0]), np.sort(b.phi[i, :n, 0]))
        assert ctx.labels[i].sum() + tgt.labels[i].sum() == b.labels[i, :n].sum()
    with pytest.raises(ValueError, match="n_context"):
        split_context_target(b, 12, seed=0)


@pytest.mark.parametrize("fraction", [None, 0.3])
def test_real_target_sampler_stays_inside_each_voxel(fraction):
    _, b = ragged()
    sampler = RealTargetSampler(
        b,
        seed=1,
        batch_size=16,
        n_events=12,
        n_context_min=3,
        n_context_max=6,
        positive_fraction=fraction,
    )
    for _ in range(5):
        context, target, weights = sampler.next()
        for record in sampler.last_provenance:
            n = COUNTS[record["trial"]]
            assert record["context"].max() < n and record["target"].max() < n
            assert not set(record["context"]) & set(record["target"])
        if fraction is not None:
            np.testing.assert_allclose(
                [(w * y).mean() for w, y in zip(weights, target.labels, strict=True)],
                [r["pool_positive_fraction"] for r in sampler.last_provenance],
            )
    with pytest.raises(ValueError):
        RealTargetSampler(b, seed=0, batch_size=2, n_events=13, n_context_min=3, n_context_max=6)


@pytest.mark.parametrize("mix_context", [False, True])
def test_mixup_pools_stay_inside_each_voxel(mix_context):
    _, b = ragged()
    source = ClassAwareMixupSource(
        b, alpha=0.2, seed=2, batch_size=16, n_events=10, n_context=4, mix_context=mix_context
    )
    for _ in range(5):
        source.next(real_target_ratio=1.0)
        for record in source.last_provenance:
            n = COUNTS[record["voxel"]]
            assert record["source_pool"].max() < n
            assert record["anchors"].max() < n and record["partners"].max() < n
            if "real_target_indices" in record:
                assert record["real_target_indices"].max() < n


def test_fixed_batch_source_respects_counts():
    _, b = ragged()
    generated = FixedBatchSource(b).generate(n_trials=20, n_events=12, seed=0)
    assert generated.n_events == 12 and not generated.is_ragged
    with pytest.raises(ValueError, match="n_events"):
        FixedBatchSource(b).generate(n_trials=2, n_events=13, seed=0)


def test_prediction_mean_and_metrics_ignore_padding():
    _, b = ragged()
    _, tgt = split_context_target(b, 4, seed=0)
    logits = np.full(tgt.labels.shape, -2.0)
    logits[~tgt.event_mask] = 50.0  # padding must not leak into any metric
    prediction = EventPrediction(logits, event_counts=tgt.event_counts)
    np.testing.assert_allclose(prediction.mean, 1 / (1 + np.exp(2.0)))

    class Fixed:
        def predict(self, target, *, context=None):
            return prediction

    metrics, arrays = evaluate_surrogate(Fixed(), tgt)
    assert metrics["events"] == int((COUNTS - 4).sum())
    assert metrics["positives"] == int(tgt.labels.sum())
    np.testing.assert_allclose(arrays["observed"], tgt.voxel_mean(tgt.labels))
    np.testing.assert_allclose(arrays["predicted"], prediction.mean)
    expected_loss = np.where(
        tgt.labels[tgt.event_mask] == 1, np.log1p(np.exp(2.0)), np.log1p(np.exp(-2.0))
    ).mean()
    assert metrics["bernoulli_log_loss"] == pytest.approx(expected_loss)
    np.testing.assert_array_equal(arrays["event_counts"], COUNTS - 4)


SAMPLING = {
    "natural": dict(sampling={"strategy": "natural"}),
    "quota": dict(
        sampling={"strategy": "positive_quota", "positive_fraction": 0.25},
        weighting={"strategy": "sampling_correction"},
    ),
    "mixup": dict(sampling={"strategy": "class_aware_mixup", "mixup": {"mix_context": True}}),
}


@pytest.mark.parametrize("kind", ["cnp", "legacy_cnp", "mlp", "transformer"])
@pytest.mark.parametrize("sampling", sorted(SAMPLING))
def test_neural_models_train_and_predict_on_ragged_data(kind, sampling):
    source, b = ragged()
    _, val = ragged(seed=5)
    spec = (
        {"kind": kind, "encoder": {"latent_dim": 8, "hidden_dims": [12]}}
        if kind in ("cnp", "legacy_cnp")
        else {"kind": kind}
    )
    model = build_surrogate(spec, source.dim_theta, source.dim_phi)
    training = NeuralTraining(
        n_steps=2,
        eval_every=1,
        batch_size=4,
        n_events=12,
        n_context_min=3,
        n_context_max=5,
        loss="theory-truth" if kind == "legacy_cnp" else "bernoulli",
        **SAMPLING[sampling],
    )
    result = model.fit(b, training, validation=Episode(*split_context_target(val, 4, seed=1)))
    assert all(np.isfinite(r["training_loss"]) for r in result.history)
    ctx, tgt = split_context_target(val, 4, seed=1)
    prediction = model.predict(tgt, context=ctx)
    np.testing.assert_array_equal(prediction.event_counts, tgt.event_counts)
    # The voxel mean equals the mean over the valid events only.
    expected = [prediction.probabilities[i, :n].mean() for i, n in enumerate(COUNTS - 4)]
    np.testing.assert_allclose(prediction.mean, expected)


def test_bdt_fits_valid_events_only():
    pytest.importorskip("sklearn")
    source, b = ragged()
    model = build_surrogate(
        {"kind": "bdt", "architecture": {"max_iter": 4, "eval_every": 2}},
        source.dim_theta,
        source.dim_phi,
    )
    result = model.fit(b, {"backend": "bdt"})
    assert result.sampling_audit["events"] == int(COUNTS.sum())
    assert result.sampling_audit["positives"] == int(b.labels.sum())
    prediction = model.predict(b)
    assert prediction.mean.shape == (4,)


def save(folder, name, batch):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = dict(mode=batch.mode.value, theta=batch.theta, phi=batch.phi, labels=batch.labels)
    if batch.event_counts is not None:
        arrays["event_counts"] = batch.event_counts
    np.savez(folder / f"{name}.npz", **arrays)


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_mfgp_stage_uses_per_voxel_target_counts(tmp_path, transform):
    pytest.importorskip("GPy")
    from core.surrogates.experiment import run_experiment
    from schemas.surrogates import SurrogateRunConfig

    root = tmp_path / "data"
    for split in ("train", "validation"):
        for i, fid in enumerate(("lf", "hf")):
            save(root / "batches" / split, fid, ragged(seed=10 * i + len(split))[1])
    config = SurrogateRunConfig(
        model={"kind": "cnp", "encoder": {"latent_dim": 8, "hidden_dims": [12]}},
        training={
            "backend": "neural",
            "n_steps": 2,
            "eval_every": 1,
            "batch_size": 2,
            "n_events": 12,
            "n_context_min": 4,
            "n_context_max": 4,
        },
        mfgp={
            "n_restarts": 1,
            "n_context": 4,
            "output_transform": transform,
            "levels": [
                {"source": "model_mean", "partition": "train", "fidelity": "lf"},
                {"source": "model_mean", "partition": "train", "fidelity": "hf"},
                {"source": "observed_fraction", "partition": "train", "fidelity": "hf"},
            ],
            "evaluation": [{"partition": "validation", "fidelity": "hf"}],
        },
        data_directory=root,
        output_directory=tmp_path / "run",
        validation_context_events=4,
    )
    out = run_experiment(config)
    # Model means cover the target events; observed fractions every event of a voxel.
    with np.load(out / "mfgp/training_arrays.npz") as data:
        np.testing.assert_array_equal(data["events_1"].ravel(), COUNTS - 4)
        np.testing.assert_array_equal(data["events_2"].ravel(), COUNTS)
        raw = data["Y_2"].ravel()
    with np.load(out / "mfgp/evaluation_0.npz") as data:
        np.testing.assert_array_equal(data["target_events"], COUNTS)
    if transform == "log":
        with np.load(out / "mfgp/fit_arrays.npz") as fit:
            np.testing.assert_allclose(fit["Y_2"].ravel(), (raw * COUNTS + 0.5) / (COUNTS + 1))


def test_rectangular_only_helpers_reject_ragged_batches():
    from core.cnp_validation import evaluate_cnp_validation
    from core.surrogate_cnp import build_cnp
    from schemas.config import EncoderConfig

    source, b = ragged()
    cnp = build_cnp(EncoderConfig(latent_dim=4, hidden_dims=[8]), source.dim_theta, source.dim_phi)
    with pytest.raises(ValueError, match="equal event counts"):
        evaluate_cnp_validation(cnp, b, n_context=4, n_mc_samples=2)
