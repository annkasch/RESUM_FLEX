"""Original-unit moments, intervals, zero handling and checkpoint semantics."""

import numpy as np
import pytest

pytest.importorskip("GPy")
from core.surrogate_mfgp import MultiFidelityGP, load_mfgp, save_mfgp
from viz.dispatch import plot_coverage_test


def test_log_moments_and_intervals(monkeypatch):
    gp = MultiFidelityGP(2, 1, output_transform="log")
    mu, var = np.array([-6.0, -4.0]), np.array([0.1, 0.4])
    monkeypatch.setattr(gp, "predict_transformed", lambda *a: (mu, var))
    x = np.zeros((2, 1))
    mean, variance = gp.predict(x)
    np.testing.assert_allclose(mean, np.exp(mu + var / 2))
    np.testing.assert_allclose(variance, np.expm1(var) * np.exp(2 * mu + var))
    lo, hi = gp.predict_interval(x, n_sigma=2)
    np.testing.assert_allclose(lo, np.exp(mu - 2 * np.sqrt(var)))
    np.testing.assert_allclose(hi, np.exp(mu + 2 * np.sqrt(var)))
    assert (lo > 0).all()


def test_log_fit_and_roundtrip(tmp_path):
    x = np.linspace(0, 1, 6)[:, None]
    ys = [np.exp(-6 + x), np.exp(-5 + x)]
    gp = MultiFidelityGP(2, 1, output_transform="log").fit([x, x], ys, n_restarts=1)
    np.testing.assert_allclose(gp.model.Y, np.concatenate([np.log(y) for y in ys]))
    restored = load_mfgp(save_mfgp(tmp_path / "gp.pkl", gp))
    for a, b in zip(gp.predict(x), restored.predict(x), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(gp.predict_interval(x), restored.predict_interval(x), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="strictly positive"):
        MultiFidelityGP(2, 1, output_transform="log").fit([x, x], [ys[0], np.zeros_like(x)])


def test_explicit_interval_coverage(tmp_path):
    obs = np.array([0.0, 0.02, 0.2])
    intervals = {k: (np.array([0.001, 0.01, 0.01]), np.array([0.1, 0.1, 0.1])) for k in (1, 2, 3)}
    coverage = plot_coverage_test(
        obs,
        np.full(3, 0.05),
        np.ones(3),
        intervals=intervals,
        out_path=tmp_path / "coverage.png",
        title="Log",
    )
    assert coverage == {f"{k}sigma": 1 / 3 for k in (1, 2, 3)}


def test_log_active_learning_rejected():
    from core.optimizer import BoxBounds, IvrAcquisition

    gp = MultiFidelityGP(2, 1, output_transform="log")
    with pytest.raises(ValueError, match="identity-space"):
        IvrAcquisition(gp, BoxBounds(low=np.array([0.0]), high=np.array([1.0])))


def test_old_checkpoint_identity_default():
    gp = MultiFidelityGP(2, 1)
    del gp.output_transform
    gp.predict_transformed = lambda *a: (np.array([0.2]), np.array([0.01]))
    np.testing.assert_array_equal(gp.predict(np.zeros((1, 1)))[0], [0.2])
