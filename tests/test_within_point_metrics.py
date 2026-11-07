"""Within-design-point log-loss gain, pair AUC, model null and feature calibration."""

import numpy as np
import pytest
from scipy.special import expit, logit

from core.surrogates.base import EventPrediction
from core.surrogates.evaluation import (
    evaluate_surrogate,
    feature_calibration,
    feature_columns,
    within_point_summary,
    within_point_terms,
)
from core.surrogates.readout import readout_metrics, within_point_null
from core.surrogates.stream_training import stream_metrics
from schemas.data_models import InputMode, StandardBatch


def phi_dependent(n_points=400, n_events=300, seed=0, event_counts=None):
    """p = rate_b (0.5 + φ0), φ0 ~ U(0, 1); φ1 is noise. Returns (batch, true p [B, N])."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    rate = rng.uniform(0.01, 0.1, n_points)  # [B]
    phi = rng.random((n_points, n_events, 2))  # [B, N, 2]
    p = rate[:, None] * (0.5 + phi[..., 0])  # [B, N]
    labels = (rng.random(p.shape) < p).astype(np.int64)
    if event_counts is not None:
        pad = np.arange(n_events)[None, :] >= np.asarray(event_counts)[:, None]
        labels[pad] = 0
    batch = StandardBatch(
        mode=InputMode.FULL,
        theta=rate[:, None],
        phi=phi,
        labels=labels,
        event_counts=event_counts,
    )
    return batch, p


def point_mean(batch, p):
    mask = batch.event_mask
    mean = np.where(mask, p, 0).sum(axis=1) / batch.events_per_voxel  # [B]
    return np.broadcast_to(mean[:, None], p.shape).copy()


def sharpen(p, factor):
    """Scale each design point's logit spread around its mean logit by ``factor``."""
    z = logit(p)
    centre = z.mean(axis=1, keepdims=True)
    return expit(centre + factor * (z - centre))


class Fixed:
    uses_context = False

    def __init__(self, probabilities):
        self.logits = logit(probabilities)

    def predict(self, target, context=None):
        return EventPrediction(self.logits, event_counts=target.event_counts)


def test_gain_matches_direct_log_losses():
    batch, p = phi_dependent(n_points=20, n_events=50)
    terms = within_point_terms(batch, batch.labels, logit(p))
    y, q = batch.labels, point_mean(batch, p)
    direct = -(y * np.log(q) + (1 - y) * np.log1p(-q)) + (y * np.log(p) + (1 - y) * np.log1p(-p))
    np.testing.assert_allclose(terms["gain"], direct.sum(axis=1), rtol=1e-10, atol=1e-12)
    kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    np.testing.assert_allclose(terms["expected"], kl.sum(axis=1), rtol=1e-10, atol=1e-12)


def test_constant_per_point_model_scores_zero_and_true_model_gains():
    batch, p = phi_dependent()
    flat = within_point_summary(
        within_point_terms(batch, batch.labels, logit(point_mean(batch, p)))
    )
    assert flat["within_point_log_loss_gain"] == pytest.approx(0, abs=1e-12)
    assert flat["within_point_log_loss_skill"] == pytest.approx(0, abs=1e-12)
    assert flat["within_point_auc"] == pytest.approx(0.5)
    assert flat["within_point_spread_z"] is None
    true = within_point_summary(within_point_terms(batch, batch.labels, logit(p)))
    # Per positive ≈ Var(0.5 + φ0) / 2 = 1/24 for small p (KL ≈ (p - p̄)² / 2p̄).
    assert true["within_point_log_loss_gain"] == pytest.approx(1 / 24, rel=0.2)
    assert true["within_point_log_loss_skill"] > 0
    assert true["within_point_auc"] > 0.55
    assert abs(true["within_point_spread_z"]) < 3


def test_spread_z_flags_over_and_underconfident_spread():
    batch, p = phi_dependent()
    wide = within_point_summary(within_point_terms(batch, batch.labels, logit(sharpen(p, 3))))
    narrow = within_point_summary(within_point_terms(batch, batch.labels, logit(sharpen(p, 0.3))))
    assert wide["within_point_spread_z"] < -3
    assert narrow["within_point_spread_z"] > 3
    # Ranking is unchanged by a monotone spread change.
    true = within_point_summary(within_point_terms(batch, batch.labels, logit(p)))
    assert wide["within_point_auc"] == pytest.approx(true["within_point_auc"])


def test_auc_matches_pair_count_and_ignores_padding():
    counts = np.array([7, 10, 4, 9])
    batch, p = phi_dependent(n_points=4, n_events=10, seed=3, event_counts=counts)
    batch = batch.model_copy(update={"labels": np.where(batch.event_mask, (p > 0.04) * 1, 0)})
    scores = logit(p)
    scores[0, :3] = scores[0, 3]  # ties count one half
    terms = within_point_terms(batch, batch.labels, scores)
    wins = pairs = 0.0
    for b, n in enumerate(counts):
        y, s = batch.labels[b, :n], scores[b, :n]
        for i in np.flatnonzero(y == 1):
            for j in np.flatnonzero(y == 0):
                wins += (s[i] > s[j]) + 0.5 * (s[i] == s[j])
                pairs += 1
    assert terms["auc_pairs"].sum() == pairs
    assert within_point_summary(terms)["within_point_auc"] == pytest.approx(wins / pairs)


def test_evaluate_surrogate_reports_within_point_metrics():
    batch, p = phi_dependent(n_points=50, n_events=100)
    metrics, arrays = evaluate_surrogate(Fixed(p), batch)
    for key in ("gain", "auc", "skill", "spread_z"):
        assert any(k.startswith("within_point_") and key in k for k in metrics)
    assert arrays["within_point_gain"].shape == (50,)


def test_null_is_seeded_and_covers_the_true_model():
    batch, p = phi_dependent(n_points=300, n_events=200, seed=5)
    terms = within_point_terms(batch, batch.labels, logit(p))
    null = within_point_null(batch, logit(p), seed=1, draws=400)
    again = within_point_null(batch, logit(p), seed=1, draws=400)
    np.testing.assert_array_equal(null[2][0], again[2][0])
    covered = ((terms["gain"] >= null[1][0]) & (terms["gain"] <= null[1][1])).mean()
    assert 0.6 < covered < 0.78
    wide = logit(sharpen(p, 3))
    wide_gain = within_point_terms(batch, batch.labels, wide)["gain"]
    wide_null = within_point_null(batch, wide, seed=1, draws=400)
    assert ((wide_gain >= wide_null[1][0]) & (wide_gain <= wide_null[1][1])).mean() < 0.5


def test_feature_calibration_finds_the_ignored_input():
    batch, p = phi_dependent(n_points=300, n_events=300, seed=7)
    names, values = feature_columns(batch)
    assert names == ["theta_0", "phi_0", "phi_1"] and values.shape == (300, 300, 3)
    mask = batch.event_mask
    y = batch.labels[mask]
    true, arrays = feature_calibration(names, values[mask], y, p[mask])
    flat, _ = feature_calibration(names, values[mask], y, point_mean(batch, p)[mask])
    assert all(true[name]["chi2_per_bin"] < 3 for name in names)
    assert flat["phi_0"]["chi2_per_bin"] > 20 and flat["phi_1"]["chi2_per_bin"] < 3
    assert arrays["feature_calibration_events"].sum(axis=1).tolist() == [mask.sum()] * 3
    assert np.all(np.diff(arrays["feature_calibration_value"], axis=1) >= 0)


def test_readout_returns_null_bands_feature_bins_and_intervals():
    batch, p = phi_dependent(n_points=60, n_events=100, seed=9)
    metrics, arrays = evaluate_surrogate(Fixed(p), batch)
    readout, extra = readout_metrics(batch, arrays, seed=2)
    assert extra["within_point_gain_null"].shape == (3, 2, 60)
    assert set(readout["within_point_gain_coverage"]) == {"1sigma", "2sigma", "3sigma"}
    assert set(readout["feature_calibration"]) == {"theta_0", "phi_0", "phi_1"}
    for key in ("within_point_log_loss_gain", "within_point_auc"):
        low, high = readout[f"{key}_ci95"]
        assert low <= readout[f"{key}_ci68"][0] <= readout[f"{key}_ci68"][1] <= high
        assert low <= metrics[key] <= high


def test_stream_metrics_calibrate_per_record_column():
    rng = np.random.default_rng(np.random.SeedSequence([11, 0]))
    features = rng.random((20000, 2))
    p = 0.02 + 0.08 * features[:, 0]
    labels = (rng.random(len(p)) < p).astype(np.int8)
    metrics, arrays = stream_metrics(labels, logit(p), reference_rate=0.06, features=features)
    assert set(metrics["feature_calibration"]) == {"record_0", "record_1"}
    assert arrays["feature_calibration_events"].shape == (2, 10)
    assert "feature_calibration" not in stream_metrics(labels, logit(p), reference_rate=0.06)[0]
    _, finer = stream_metrics(
        labels, logit(p), reference_rate=0.06, features=features, feature_bins=40
    )
    assert finer["feature_calibration_events"].shape == (2, 40)


def test_readout_feature_bins_and_config():
    from pydantic import ValidationError

    from schemas.surrogates import FeatureCalibrationConfig

    batch, p = phi_dependent(n_points=60, n_events=100, seed=9)
    _, arrays = evaluate_surrogate(Fixed(p), batch)
    _, extra = readout_metrics(batch, arrays, seed=2, feature_bins=25, null_draws=200)
    assert extra["feature_calibration_events"].shape == (3, 25)
    reference = within_point_null(batch, arrays["logits"], seed=2, draws=200)
    assert np.array_equal(extra["within_point_gain_null"][2, 0], reference[3][0])  # 200 draws
    assert extra["feature_calibration_events"].sum(axis=1).tolist() == [6000] * 3
    assert FeatureCalibrationConfig().bins == 10
    with pytest.raises(ValidationError):
        FeatureCalibrationConfig(bins=1)
    from schemas.surrogates import SurrogateRunConfig

    base = dict(model={"kind": "mlp"}, training={"backend": "neural"}, data_directory="d")
    assert SurrogateRunConfig(**base, output_directory="o").within_point_null_draws == 1000
    with pytest.raises(ValidationError):
        SurrogateRunConfig(**base, output_directory="o", within_point_null_draws=10)


def test_plots_render(tmp_path):
    from viz.surrogate import plot_feature_calibration, plot_within_point_gain

    batch, p = phi_dependent(n_points=60, n_events=100, seed=13)
    _, arrays = evaluate_surrogate(Fixed(sharpen(p, 2)), batch)
    arrays.update(readout_metrics(batch, arrays, seed=2)[1])
    plot_within_point_gain(arrays, tmp_path / "gain.png", title="test")
    plot_feature_calibration(arrays, tmp_path / "features.png", title="test")
    assert (tmp_path / "gain.png").stat().st_size > 0
    assert (tmp_path / "features.png").stat().st_size > 0
