"""Residual learnability test: cross-fitted LightGBM correction of the model logit."""

import sys

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import logit

from core.surrogates.evaluation import evaluate_surrogate, feature_columns, within_point_terms
from core.surrogates.readout import readout_metrics
from core.surrogates.residual import (
    balanced_folds,
    correction_bins,
    design_point_residual,
    require_lightgbm,
    residual_test,
    stream_residual,
    within_gain,
)
from core.surrogates.stream_training import stream_metrics
from schemas.surrogates import ResidualTestConfig, SurrogateRunConfig
from tests.test_within_point_metrics import Fixed, phi_dependent, point_mean

SETTINGS = ResidualTestConfig()


def flat_inputs(batch, probabilities):
    """Valid-event features, labels, logits and design-point index of a batch."""
    mask = batch.event_mask
    _, values = feature_columns(batch)
    return values[mask], batch.labels[mask], logit(probabilities)[mask], np.nonzero(mask)[0]


@pytest.fixture(scope="module")
def data():
    pytest.importorskip("lightgbm")
    return phi_dependent(n_points=200, n_events=300)


def test_balanced_folds_spread_positives_and_are_seeded():
    positives = np.random.default_rng(0).poisson(5, 103)
    folds = balanced_folds(positives, 5, np.random.default_rng(1))
    totals = np.bincount(folds, positives, minlength=5)
    assert set(folds) == set(range(5))
    assert totals.max() - totals.min() <= positives.max()
    assert np.array_equal(folds, balanced_folds(positives, 5, np.random.default_rng(1)))


def test_within_gain_matches_within_point_terms(data):
    batch, p = data
    q = np.clip(p * 1.3, 1e-6, 0.99)
    features, labels, z, groups = flat_inputs(batch, q)
    terms = within_point_terms(batch, batch.labels, logit(q))
    assert np.allclose(within_gain(groups, labels, z), terms["gain"])


def test_true_probabilities_give_no_gain(data):
    batch, p = data
    metrics, _ = residual_test(*flat_inputs(batch, p), SETTINGS, seed=1)
    assert abs(metrics["gain"]) < 0.005
    assert metrics["folds"] == 5 and len(metrics["trees"]) == 5


def test_missing_phi_dependence_is_the_phi_part(data):
    batch, p = data
    metrics, arrays = residual_test(*flat_inputs(batch, point_mean(batch, p)), SETTINGS, seed=1)
    # p ∝ 0.5 + φ0 with φ0 ~ U(0, 1): the ideal gain is ≈ 1/24 nats per positive.
    assert 0.025 < metrics["phi_gain"] < 0.07
    assert metrics["gain_ci95"][0] > 0
    for key in ("gain", "phi_gain", "rate_gain"):
        low, high = metrics[f"{key}_ci95"]
        assert low < metrics[f"{key}_ci68"][0] < metrics[f"{key}_ci68"][1] < high
    assert abs(metrics["rate_gain"]) < 0.005
    assert np.isclose(metrics["gain"], metrics["phi_gain"] + metrics["rate_gain"])
    positives = batch.labels[batch.event_mask].sum()
    assert np.isclose(arrays["residual_gain_per_point"].sum() / positives, metrics["gain"])


def test_rate_error_is_the_rate_part(data):
    batch, p = data
    metrics, _ = residual_test(*flat_inputs(batch, p * 1.5), SETTINGS, seed=1)
    assert metrics["rate_gain"] > 0.05
    assert abs(metrics["phi_gain"]) < 0.01


def test_seeded_and_skipped_below_min_positives(data):
    batch, p = data
    inputs = flat_inputs(batch, point_mean(batch, p))
    first, a = residual_test(*inputs, SETTINGS, seed=3)
    second, b = residual_test(*inputs, SETTINGS, seed=3)
    assert first == second and np.array_equal(a["residual_logits"], b["residual_logits"])
    strict = ResidualTestConfig(min_positives=10**6)
    skipped, arrays = residual_test(*inputs, strict, seed=3)
    assert skipped["gain"] is None and skipped["skipped"] and arrays == {}


def test_readout_reports_residual_test_with_padding(data):
    batch, p = phi_dependent(n_points=60, n_events=200, event_counts=np.full(60, 150))
    model = Fixed(point_mean(batch, p))
    _, arrays = evaluate_surrogate(model, batch)
    settings = ResidualTestConfig(min_positives=10, min_data_in_leaf=50)
    readout, extra = readout_metrics(batch, arrays, seed=2, residual=settings)
    assert readout["residual_test"]["units"] == "design_points"
    assert readout["residual_test"]["gain"] is not None
    assert extra["residual_logits"].shape == batch.labels.shape
    assert np.isnan(extra["residual_logits"][:, 150:]).all()
    assert (extra["residual_fold"][:, 150:] == -1).all()
    assert "residual_test" not in readout_metrics(batch, arrays, seed=2)[0]


def test_stream_metrics_split_by_event(data):
    batch, p = data
    features, labels, z, _ = flat_inputs(batch, point_mean(batch, p))
    metrics, arrays = stream_metrics(
        labels, z, reference_rate=0.06, features=features, residual=SETTINGS, seed=1
    )
    result = metrics["residual_test"]
    assert result["units"] == "events" and result["phi_gain"] is None
    assert result["gain_ci95"][0] > 0
    assert arrays["residual_logits"].dtype == np.float32


def test_stream_subsample_keeps_about_max_positives(data):
    batch, p = data
    features, labels, z, _ = flat_inputs(batch, point_mean(batch, p))
    settings = ResidualTestConfig(max_positives=1000)
    metrics, arrays = stream_residual(features, labels, z, settings, seed=1)
    assert metrics["positives_available"] == labels.sum() > 2000
    assert metrics["events_available"] == len(labels)
    assert np.isclose(metrics["subsample_fraction"], 1000 / labels.sum())
    assert 850 < metrics["positives"] < 1150
    kept = ~np.isnan(arrays["residual_logits"])
    assert kept.sum() == metrics["events"] and (arrays["residual_fold"][~kept] == -1).all()
    assert metrics["gain_ci95"][0] > 0  # the missing φ0 trend is still found
    again, _ = stream_residual(features, labels, z, settings, seed=1)
    assert again == metrics
    every, _ = stream_residual(features, labels, z, ResidualTestConfig(max_positives=None), seed=1)
    assert every["subsample_fraction"] == 1.0 and every["events"] == len(labels)


def test_correction_bins_count_every_event_once():
    rng = np.random.default_rng(np.random.SeedSequence([5, 0]))
    features = rng.random((500, 2))  # [E, F]
    z = rng.normal(-3, 1, 500)
    labels = (rng.random(500) < 0.05).astype(np.int64)
    arrays = correction_bins(["a", "b"], features, labels, z, z, n_bins=7)
    assert list(arrays["residual_correction_names"]) == ["logit_p", "a", "b"]
    assert arrays["residual_correction_events"].shape == (3, 7)
    assert (arrays["residual_correction_events"].sum(axis=1) == 500).all()
    assert (arrays["residual_correction_hits"].sum(axis=1) == labels.sum()).all()
    assert np.allclose(
        arrays["residual_correction_predicted"], arrays["residual_correction_corrected"]
    )
    assert (np.diff(arrays["residual_correction_value"], axis=1) >= 0).all()
    with pytest.raises(ValueError, match="one name per column"):
        correction_bins(["a"], features, labels, z, z)


def test_correction_follows_the_missing_phi_trend(data):
    batch, p = data
    q = point_mean(batch, p)
    _, arrays = design_point_residual(batch, batch.labels, logit(q), SETTINGS, seed=1, bins=5)
    names = list(arrays["residual_correction_names"])
    assert names == ["logit_p", "theta_0", "phi_0", "phi_1"]
    ratio = np.log(
        arrays["residual_correction_corrected"] / arrays["residual_correction_predicted"]
    )
    observed = np.log(arrays["residual_correction_hits"] / arrays["residual_correction_predicted"])
    # True / model rate = 0.5 + φ0: log ratio from ≈ log 0.6 to ≈ log 1.4 over the φ0 bins.
    assert ratio[2, 0] < -0.25 and ratio[2, -1] > 0.15
    assert np.abs(ratio[2] - observed[2]).max() < 0.1
    assert np.abs(ratio[3]).max() < 0.1  # φ1 is noise


def test_stream_correction_covers_kept_events(data):
    batch, p = data
    features, labels, z, _ = flat_inputs(batch, point_mean(batch, p))
    metrics, arrays = stream_residual(
        features, labels, z, ResidualTestConfig(max_positives=1000), seed=1, bins=4
    )
    assert list(arrays["residual_correction_names"]) == [
        "logit_p",
        "phi_0",
        "phi_1",
        "phi_2",
    ]
    assert (arrays["residual_correction_events"].sum(axis=1) == metrics["events"]).all()


def test_correction_plot_renders(data, tmp_path):
    from viz.surrogate import plot_residual_correction

    batch, p = data
    _, arrays = design_point_residual(
        batch, batch.labels, logit(point_mean(batch, p)), SETTINGS, seed=1, bins=40
    )
    arrays["residual_correction_hits"][1, 0] = 0  # an empty bin is drawn as an upper limit
    plot_residual_correction(arrays, tmp_path / "correction.png", title="test")
    assert (tmp_path / "correction.png").stat().st_size > 0


def test_max_positives_only_for_stream_records():
    base = dict(
        model={"kind": "mlp"},
        training={"backend": "neural"},
        data_directory="d",
        output_directory="o",
    )
    SurrogateRunConfig(**base, residual_test={"max_positives": 20_000})  # default: accepted
    with pytest.raises(ValidationError, match="stream validation events only"):
        SurrogateRunConfig(**base, residual_test={"max_positives": 5000})


def test_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ResidualTestConfig(fold=3)
    with pytest.raises(ValidationError):
        ResidualTestConfig(folds=1)


def test_missing_lightgbm_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(ImportError, match="--extra lgbm"):
        require_lightgbm()
