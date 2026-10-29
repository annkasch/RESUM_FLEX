"""Per-voxel z-score plots for event models and GPs."""

import numpy as np
import pytest

from viz.zscore import gp_zscores, plot_zscores


def test_plot_zscores_summary_and_files(tmp_path):
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    summary = plot_zscores(z, rng.uniform(1e-3, 1e-2, 400), tmp_path / "z", title="t")
    assert abs(summary["mean"]) < 0.15 and abs(summary["std"] - 1) < 0.1
    assert summary["voxels"] == 400 and 0 < summary["outside_2"] < 0.1
    assert (tmp_path / "z.png").exists() and (tmp_path / "z.pdf").exists()
    with pytest.raises(ValueError, match="finite"):
        plot_zscores(np.array([np.nan]), np.array([1.0]), tmp_path / "bad", title="t")


@pytest.mark.parametrize("log", [False, True])
def test_gp_zscores_recover_mean_and_sigma(log):
    mu, sigma = np.array([-6.0, -5.5]), np.array([0.2, 0.4])  # GP space
    n = np.array([5000.0, 5000.0])
    lower, upper = mu - sigma, mu + sigma
    if log:
        observed = (np.exp(mu + sigma) * (n + 1) - 0.5) / n  # smoothed fraction at +1σ
        lower, upper = np.exp(lower), np.exp(upper)
    else:
        mu, sigma = np.array([0.002, 0.004]), np.array([0.0005, 0.001])
        lower, upper = mu - sigma, mu + sigma
        observed = mu - 2 * sigma
    arrays = dict(lower_1=lower, upper_1=upper, observed=observed, target_events=n,
                  output_transform=np.array("log" if log else "identity"))
    np.testing.assert_allclose(gp_zscores(arrays, 0.5), [1, 1] if log else [-2, -2])
