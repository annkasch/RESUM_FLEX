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
    assert set(row["coverage"]) == {"1", "2", "3"}
    assert all(0 <= c <= 1 for c in row["coverage"].values())
    log_terms = {"pseudocount_shift", "jensen_shift", "count_log_variance", "back_conversion_shift"}
    assert log_terms <= set(row) if transform == "log" else not log_terms & set(row)


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
    from core.mfgp_pipeline import fit_mfgp, fit_mfgp_log_corrected, log_target_offset

    X, Y, events = _two_levels()
    np.random.seed(0)
    gp, offsets, info = fit_mfgp_log_corrected(
        X, Y, events=[None, events], max_refits=8, pseudocount=0.5, n_restarts=1
    )
    assert offsets[0] is None and offsets[1].shape == (12,)
    assert 1 <= info["refits"] <= 8 and info["converged"] == (info["max_change"] <= 0.01)
    np.testing.assert_allclose(gp.model.Y[30:, 0], np.log(Y[1][:, 0]) - offsets[1])
    if info["converged"]:  # offsets are (to tolerance) those of the final rate
        rate, _ = gp.predict(X[1], fidelity=1)
        np.testing.assert_allclose(offsets[1], log_target_offset(rate, events, 0.5), atol=0.02)
    np.random.seed(0)
    plain, none, info0 = fit_mfgp_log_corrected(
        X, Y, events=[None, events], max_refits=0, pseudocount=0.5, n_restarts=1
    )
    assert none == [None, None] and info0["refits"] == 0
    np.random.seed(0)
    reference = fit_mfgp(X, Y, output_transform="log", n_restarts=1)
    np.testing.assert_allclose(plain.model.param_array, reference.model.param_array)
    with pytest.raises(ValueError, match="one entry"):
        fit_mfgp_log_corrected(X, Y, events=[events], max_refits=1, pseudocount=0.5)


def test_log_bias_refits_config():
    from core.mfgp_pipeline import resolve_log_bias_refits
    from schemas.surrogates import MFGPStageConfig

    levels = [
        {"source": "model_mean", "partition": "train", "fidelity": "lf"},
        {"source": "observed_fraction", "partition": "train", "fidelity": "hf"},
    ]
    assert MFGPStageConfig(levels=levels).log_bias_max_refits is None
    MFGPStageConfig(levels=levels, output_transform="log", log_bias_max_refits=0)
    MFGPStageConfig(levels=levels, log_bias_max_refits=0)
    with pytest.raises(ValueError, match="log only"):
        MFGPStageConfig(levels=levels, log_bias_max_refits=3)
    assert resolve_log_bias_refits(None, "log") == 8
    assert resolve_log_bias_refits(None, "identity") == 0
    with pytest.raises(ValueError):
        resolve_log_bias_refits(2, "identity")
