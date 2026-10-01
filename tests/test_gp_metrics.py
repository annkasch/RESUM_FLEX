import json

import numpy as np
import pytest

from core.gp_metrics import gp_metrics, mfgp_metrics


def test_bias_trend_and_json_null():
    result = gp_metrics([1, 2, 3], [2, 3, 4])
    assert result["mean_bias"] == 1
    assert result["mae"] == 1
    assert result["rmse"] == 1
    assert result["pearson_r"] == pytest.approx(1)
    assert result["spearman_r"] == pytest.approx(1)
    assert result["calibration_intercept"] == pytest.approx(-1)
    assert result["calibration_slope"] == pytest.approx(1)
    constant = gp_metrics([0, 0], [0, 0])
    assert constant["pearson_r"] is None
    assert constant["calibration_slope"] is None
    json.dumps(constant, allow_nan=False)


def test_crps_and_interval_scores_against_definition():
    y = np.array([0.0, 4.0])
    samples = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    result = gp_metrics(
        y,
        samples.mean(0),
        samples=samples,
        intervals={1: (np.array([0.0, 1.0]), np.array([2.0, 3.0]))},
    )
    expected = np.abs(samples - y).mean() - 0.5 * np.abs(samples[:, None] - samples[None, :]).mean()
    assert result["crps"] == pytest.approx(expected)
    d = result["interval_metrics"]["1"]
    assert d["inside"] == 1
    assert d["coverage"] == 0.5
    assert d["mean_width"] == 2
    alpha = 1 - d["nominal_probability"]
    assert d["interval_score"] == pytest.approx(2 + 1 / alpha)
    expected_wis = (
        0.5 * np.mean(np.abs(y - np.median(samples, axis=0))) + alpha / 2 * d["interval_score"]
    ) / 1.5
    assert result["weighted_interval_score"] == pytest.approx(expected_wis)


def test_invalid_metrics_fail():
    with pytest.raises(ValueError):
        gp_metrics([0, np.nan], [0, 1])
    with pytest.raises(ValueError):
        gp_metrics([0], [0], intervals={1: ([1], [0])})


class LogGP:
    output_transform = "log"

    def predict_transformed(self, theta, fidelity=None):
        return np.zeros(len(theta)), np.ones(len(theta))

    def predict(self, theta, fidelity=None):
        return np.full(len(theta), np.exp(0.5)), np.full(len(theta), (np.e - 1) * np.e)

    def predict_interval(self, theta, fidelity=None, n_sigma=1):
        return np.full(len(theta), np.exp(-n_sigma)), np.full(len(theta), np.exp(n_sigma))


def test_log_density_support_and_original_units():
    finite = mfgp_metrics(LogGP(), np.zeros((2, 1)), np.ones(2))
    assert finite["predictive_nll"] == pytest.approx(0.5 * np.log(2 * np.pi))
    result = mfgp_metrics(LogGP(), np.zeros((2, 1)), np.array([0.0, 1.0]))
    assert result["predictive_nll"] is None
    assert "outside support" in result["predictive_nll_status"]
    assert result["interval_metrics"]["3"]["inside"] == 1
    json.dumps(result, allow_nan=False)
