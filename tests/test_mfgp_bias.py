"""Exact log-target bias terms, shared GP target preparation and the synthetic noise study."""

import numpy as np
import pytest

from core.mfgp_pipeline import fit_targets


def test_fit_targets_identity_and_log():
    values, events = np.array([0.0, 0.1]), np.array([10.0, 20.0])
    for source in ("observed_fraction", "model_mean"):
        out = fit_targets(values, events, source=source, output_transform="identity")
        np.testing.assert_array_equal(out, values)
    smoothed = fit_targets(
        values, events, source="observed_fraction", output_transform="log", pseudocount=0.5
    )
    np.testing.assert_allclose(smoothed, [0.5 / 11, 2.5 / 21])
    floored = fit_targets(values, events, source="model_mean", output_transform="log", floor=1e-6)
    np.testing.assert_array_equal(floored, [1e-6, 0.1])
    with pytest.raises(ValueError, match="nonnegative"):
        fit_targets(-values - 1, events, source="model_mean", output_transform="log")
    with pytest.raises(ValueError, match="source"):
        fit_targets(values, events, source="other", output_transform="log")


def test_log_target_terms_exact_and_vanishing():
    pytest.importorskip("GPy")
    from core.mfgp_bias import log_target_terms

    p, n, a = np.array([0.002, 0.01, 0.01]), np.array([2500.0, 500.0, 1e5]), 0.5
    terms = log_target_terms(p, n, a)
    np.testing.assert_allclose(terms["pseudocount_shift"], (n * p + a) / ((n + 2 * a) * p) - 1)
    rng = np.random.default_rng(np.random.SeedSequence([0, 0]))
    m = rng.binomial(n.astype(int), p, size=(200_000, 3))
    log_y = np.log((m + a) / (n + 2 * a))
    mean_y = (n * p + a) / (n + 2 * a)
    np.testing.assert_allclose(
        terms["jensen_shift"], np.exp(log_y.mean(0) - np.log(mean_y)) - 1, atol=2e-3
    )
    np.testing.assert_allclose(terms["log_variance"], log_y.var(0), rtol=2e-2)
    # 5 expected hits: pseudocount up, Jensen down; 1000 hits: both negligible.
    assert terms["pseudocount_shift"][0] > 0.05 and terms["jensen_shift"][0] < -0.05
    assert abs(terms["pseudocount_shift"][2]) < 1e-3 and abs(terms["jensen_shift"][2]) < 1e-3
    with pytest.raises(ValueError):
        log_target_terms(np.array([0.0]), np.array([10.0]), a)


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_noise_study_runs_and_is_seeded(transform):
    pytest.importorskip("GPy")
    from core.mfgp_bias import noise_study

    kwargs = dict(output_transform=transform, n_low=40, n_top=15, n_test=30, n_restarts=1)
    row = noise_study(5, seed=3, **kwargs)
    assert row == noise_study(5, seed=3, **kwargs)
    assert row["events"] > 0 and row["mean_hits_dim"] < 5
    for key in ("rate_bias", "rate_bias_dim", "rate_bias_bright", "top_noise_variance"):
        assert np.isfinite(row[key])
    for key in ("coverage", "coverage_dim", "coverage_bright"):
        assert set(row[key]) == {"1", "2", "3"}
        assert all(0 <= c <= 1 for c in row[key].values())
    log_terms = {"pseudocount_shift", "jensen_shift", "count_log_variance", "back_conversion_shift"}
    assert log_terms <= set(row) if transform == "log" else not log_terms & set(row)
    if transform == "log":
        binomial = noise_study(5, seed=3, observation_noise="binomial", **kwargs)
        assert binomial["refits"]["point_noise"] and binomial["refits"]["bias_correction"]
    else:
        binomial = noise_study(50, seed=3, observation_noise="binomial", **kwargs)
        assert binomial["refits"]["point_noise"] and not binomial["refits"]["bias_correction"]


def test_sweep_script_writes_rows(tmp_path):
    pytest.importorskip("GPy")
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts/mfgp_noise_sweep.py"
    out = tmp_path / "rows.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--hits",
            "50",
            "--transforms",
            "log",
            "--n-restarts",
            "1",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    (row,) = json.loads(out.read_text())
    assert row["output_transform"] == "log" and row["hits"] == 50


def test_log_target_offset_is_pseudocount_plus_jensen():
    pytest.importorskip("GPy")
    from core.mfgp_bias import log_target_terms
    from core.mfgp_pipeline import log_target_offset

    p, n = np.array([0.001, 0.01, 0.3]), np.array([1000.0, 500.0, 50.0])
    terms = log_target_terms(p, n, 0.5)
    np.testing.assert_allclose(
        log_target_offset(p, n, 0.5),
        np.log1p(terms["pseudocount_shift"]) + np.log1p(terms["jensen_shift"]),
    )


def _two_levels():
    rng = np.random.default_rng(np.random.SeedSequence([5, 0]))
    x_low, x_top = rng.uniform(size=(30, 1)), rng.uniform(size=(12, 1))
    p_top = 0.02 * np.exp(np.sin(3 * x_top[:, 0]))
    events = np.full(12, 200.0)
    fraction = rng.binomial(200, p_top) / 200
    low = 0.02 * np.exp(np.sin(3 * x_low))
    top = fit_targets(fraction, events, source="observed_fraction", output_transform="log")
    return [x_low, x_top], [low, top[:, None]], events


def test_log_corrected_fit_offsets_and_stopping():
    pytest.importorskip("GPy")
    from core.mfgp_pipeline import fit_mfgp, fit_mfgp_corrected, log_target_offset

    X, Y, events = _two_levels()
    np.random.seed(0)
    result = fit_mfgp_corrected(
        X,
        Y,
        events=[None, events],
        output_transform="log",
        max_refits=8,
        pseudocount=0.5,
        n_restarts=1,
    )
    gp, offsets = result.gp, result.offsets
    assert offsets[0] is None and offsets[1].shape == (12,)
    assert result.point_noise == [None, None]
    assert 1 <= result.refits <= 8 and result.converged == (result.max_change <= 0.01)
    np.testing.assert_allclose(gp.model.Y[30:, 0], np.log(Y[1][:, 0]) - offsets[1])
    if result.converged:  # offsets are (to tolerance) those of the final rate
        rate, _ = gp.predict(X[1], fidelity=1)
        np.testing.assert_allclose(offsets[1], log_target_offset(rate, events, 0.5), atol=0.02)
    np.random.seed(0)
    plain = fit_mfgp_corrected(
        X,
        Y,
        events=[None, events],
        output_transform="log",
        max_refits=0,
        pseudocount=0.5,
        n_restarts=1,
    )
    assert plain.offsets == [None, None] and plain.refits == 0
    np.random.seed(0)
    reference = fit_mfgp(X, Y, output_transform="log", n_restarts=1)
    np.testing.assert_allclose(plain.gp.model.param_array, reference.model.param_array)
    with pytest.raises(ValueError, match="one entry"):
        fit_mfgp_corrected(
            X, Y, events=[events], output_transform="log", max_refits=1, pseudocount=0.5
        )
    with pytest.raises(ValueError, match="refit"):
        fit_mfgp_corrected(
            X,
            Y,
            events=[None, events],
            output_transform="log",
            max_refits=0,
            pseudocount=0.5,
            point_noise=True,
        )


def test_log_corrected_fit_with_point_noise():
    pytest.importorskip("GPy")
    from core.mfgp_pipeline import binomial_log_moments, count_noise, fit_mfgp_corrected

    X, Y, events = _two_levels()
    np.random.seed(0)
    result = fit_mfgp_corrected(
        X,
        Y,
        events=[None, events],
        output_transform="log",
        max_refits=8,
        pseudocount=0.5,
        n_restarts=1,
        point_noise=True,
    )
    gp, noise = result.gp, result.point_noise[1]
    assert result.point_noise[0] is None and noise.shape == (12,) and (noise > 0).all()
    assert result.summary()["point_noise"] is True
    # The training rows carry the fixed noise; the learned level noise comes on top.
    fixed = gp.model.Y_metadata["point_noise"]
    np.testing.assert_array_equal(fixed[:30], 0)
    np.testing.assert_allclose(fixed[30:], noise)
    level = gp.model.likelihood.gaussian_variance(gp.model.Y_metadata)
    learned = [float(lik.variance[0]) for lik in gp.model.likelihood.likelihoods_list]
    np.testing.assert_allclose(level[30:], learned[1] + noise)
    rate, _ = gp.predict(X[1], fidelity=1)
    np.testing.assert_allclose(
        count_noise(rate, events, output_transform="log", pseudocount=0.5),
        binomial_log_moments(rate, events, 0.5)[1],
    )
    with pytest.raises(ValueError, match="outside"):
        count_noise(np.array([1.2]), np.array([10.0]), output_transform="identity")


def test_predict_adds_point_noise():
    pytest.importorskip("GPy")
    from core.mfgp_pipeline import fit_mfgp

    X, Y, _ = _two_levels()
    np.random.seed(0)
    gp = fit_mfgp(X, Y, output_transform="log", n_restarts=1, point_noise=[None, np.full(12, 0.05)])
    x = X[1][:4]
    mu, total = gp.predict_transformed(x)
    noise = np.full(4, 0.2)
    mu2, total2 = gp.predict_transformed(x, noise=noise)
    np.testing.assert_allclose(mu2, mu)
    np.testing.assert_allclose(total2, total + noise)
    mean, _ = gp.predict(x)
    mean2, variance2 = gp.predict(x, noise=noise)
    np.testing.assert_allclose(mean2, mean)  # noise widens, never shifts the rate
    np.testing.assert_allclose(variance2, np.expm1(total2) * np.exp(2 * mu + total2))
    lo, hi = gp.predict_interval(x, n_sigma=1, noise=noise)
    np.testing.assert_allclose(hi / lo, np.exp(2 * np.sqrt(total2)))
    with pytest.raises(ValueError, match="nonnegative"):
        gp.predict(x, noise=-noise)
    with pytest.raises(ValueError, match="point_noise"):
        fit_mfgp(X, Y, output_transform="log", n_restarts=1, point_noise=[None])


def test_refit_and_noise_config():
    from core.mfgp_pipeline import resolve_max_refits
    from schemas.surrogates import MFGPStageConfig

    levels = [
        {"source": "model_mean", "partition": "train", "fidelity": "lf"},
        {"source": "observed_fraction", "partition": "train", "fidelity": "hf"},
    ]
    assert MFGPStageConfig(levels=levels).max_refits is None
    MFGPStageConfig(levels=levels, output_transform="log", max_refits=0)
    MFGPStageConfig(levels=levels, max_refits=0)
    with pytest.raises(ValueError, match="max_refits needs"):
        MFGPStageConfig(levels=levels, max_refits=3)
    for transform in ("identity", "log"):
        MFGPStageConfig(levels=levels, output_transform=transform, observation_noise="binomial")
        with pytest.raises(ValueError, match="max_refits > 0"):
            MFGPStageConfig(
                levels=levels,
                output_transform=transform,
                observation_noise="binomial",
                max_refits=0,
            )
    with pytest.raises(ValueError, match="observed_fraction level"):
        MFGPStageConfig(levels=levels[:1] * 2, observation_noise="binomial")
    assert resolve_max_refits(None, "log") == 8
    assert resolve_max_refits(None, "identity") == 0
    assert resolve_max_refits(None, "identity", "binomial") == 8
    with pytest.raises(ValueError):
        resolve_max_refits(2, "identity")
    with pytest.raises(ValueError):
        resolve_max_refits(0, "identity", "binomial")


def test_identity_point_noise_from_predicted_rate():
    pytest.importorskip("GPy")
    from core.mfgp_pipeline import fit_mfgp_corrected

    X, _, events = _two_levels()
    rng = np.random.default_rng(np.random.SeedSequence([5, 1]))
    p_top = 0.02 * np.exp(np.sin(3 * X[1][:, 0]))
    Y = [0.02 * np.exp(np.sin(3 * X[0])), (rng.binomial(200, p_top) / 200)[:, None]]
    np.random.seed(0)
    result = fit_mfgp_corrected(
        X,
        Y,
        events=[None, events],
        output_transform="identity",
        max_refits=8,
        n_restarts=1,
        point_noise=True,
    )
    assert result.offsets == [None, None]  # no bias correction in identity mode
    assert result.refits >= 2  # the first refit only sets the noise
    np.testing.assert_allclose(result.gp.model.Y[30:], Y[1])  # targets unchanged
    np.testing.assert_allclose(
        result.gp.model.Y_metadata["point_noise"][30:], result.point_noise[1]
    )
    assert result.summary()["point_noise"] and not result.summary()["bias_correction"]
    with pytest.raises(ValueError, match="only for point_noise"):
        fit_mfgp_corrected(X, Y, events=[None, events], output_transform="identity", max_refits=2)
    # A predicted rate <= 0 fails instead of being clipped.
    with pytest.raises(ValueError, match="outside"):
        fit_mfgp_corrected(
            X,
            [Y[0], Y[1] - 0.05],
            events=[None, events],
            output_transform="identity",
            max_refits=2,
            n_restarts=1,
            point_noise=True,
        )
