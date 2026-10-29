"""Tests for the level-based MFGP pipeline helpers.

The synthetic wrappers (``synthetic_levels``, ``evaluate_mfgp_coverage``) get end-to-end
coverage in ``test_mfgp_recovery``; this file pins the batch-based entry points: shapes,
shared targets, mode rejection, and parity with the wrappers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from core import (
    BatchLevel,
    build_cnp,
    evaluate_mfgp_coverage_from_batch,
    fit_mfgp,
    levels_from_batches,
    observed_fraction,
    synthetic_levels,
    train_cnp,
)
from data import for_scenario
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"GPy.*")

PAPER_LEVELS = [("model_mean", "lf"), ("model_mean", "hf"), ("observed_fraction", "hf")]


@pytest.mark.parametrize("n_levels", [2, 3, 4])
def test_fit_any_number_of_levels(n_levels) -> None:
    x = np.linspace(-1.0, 1.0, 9)[:, None]
    X_list = [x[:: 2**k] for k in range(n_levels)]
    Y_list = [xs**2 + 0.01 * k for k, xs in enumerate(X_list)]
    model = fit_mfgp(X_list, Y_list, kernel="matern52", n_restarts=1)
    assert model.n_fidelities == n_levels and model.kernel_name == "matern52"
    with pytest.raises(ValueError, match="at least two levels"):
        fit_mfgp(X_list[:1], Y_list[:1])


def _quick_cnp(name: str = "S7"):
    """Cheap CNP for shape-level tests — 200 steps, tiny encoder."""
    torch.manual_seed(0)
    np.random.seed(0)
    gen = for_scenario(name, seed=0)
    enc = EncoderConfig(type="mlp", latent_dim=16, hidden_dims=[32, 32])
    cnp = build_cnp(enc, gen.dim_theta, gen.dim_phi)
    train_cnp(
        cnp,
        gen,
        cnp_config=CNPConfig(
            n_context_min=16,
            n_context_max=64,
        ),
        training_config=TrainingConfig(
            n_steps=200,
            learning_rate=1.0e-3,
            batch_size=16,
            n_events_per_trial=64,
            eval_every=0,
            seed=0,
        ),
    )
    return gen, cnp


def test_levels_from_batches_shapes_and_shared_targets() -> None:
    gen, cnp = _quick_cnp()
    lf = gen.generate(n_trials=20, n_events=32, seed=1)
    hf = gen.generate(n_trials=5, n_events=64, seed=2)
    X_list, Y_list = levels_from_batches(
        cnp,
        [
            BatchLevel("model_mean", lf, n_context=16, seed=1),
            BatchLevel("model_mean", hf, n_context=32, seed=101),
            BatchLevel("observed_fraction", hf, n_context=32, seed=101),
        ],
        n_mc_samples=8,
    )
    d = gen.dim_theta
    assert [x.shape for x in X_list] == [(20, d), (5, d), (5, d)]
    assert [y.shape for y in Y_list] == [(20, 1), (5, 1), (5, 1)]
    for y in Y_list:
        assert np.all((y >= 0) & (y <= 1))
    # The observed fraction covers the same target events as the HF model mean.
    fraction, events = observed_fraction(hf, n_context=32, seed=101)
    np.testing.assert_array_equal(Y_list[2].ravel(), fraction)
    assert (events == 32).all()
    with pytest.raises(ValueError, match="need an event model"):
        levels_from_batches(None, [BatchLevel("model_mean", lf, n_context=16)])
    with pytest.raises(ValueError, match="n_context"):
        levels_from_batches(cnp, [BatchLevel("model_mean", lf)])


def test_batch_levels_reject_event_only_and_unknown_source() -> None:
    other = for_scenario("S5", seed=0)  # EVENT_ONLY
    bad_batch = other.generate(n_trials=3, n_events=16, seed=0)
    with pytest.raises(ValueError, match="must contain θ"):
        BatchLevel("observed_fraction", bad_batch)
    good = for_scenario("S7", seed=0).generate(n_trials=3, n_events=16, seed=0)
    with pytest.raises(ValueError, match="source must be one of"):
        BatchLevel("raw", good)


def _paper_mfgp(gen, cnp, n_hf_trials=8):
    X_list, Y_list = synthetic_levels(
        cnp,
        gen,
        PAPER_LEVELS,
        n_trials={"lf": 40, "hf": n_hf_trials},
        n_events={"lf": 32, "hf": 64},
        seed=0,
    )
    return fit_mfgp(X_list, Y_list, n_restarts=2)


def test_evaluate_coverage_from_batch_shapes_and_keys() -> None:
    gen, cnp = _quick_cnp()
    mfgp = _paper_mfgp(gen, cnp)
    holdout = gen.generate(n_trials=15, n_events=64, seed=999)
    result = evaluate_mfgp_coverage_from_batch(mfgp, holdout, n_context=32, seed=999)
    assert result["theta"].shape == (15, gen.dim_theta)
    assert result["y_obs"].shape == (15,)
    assert result["mu"].shape == (15,)
    assert result["sigma"].shape == (15,)
    for key in ("1sigma", "2sigma", "3sigma"):
        assert 0.0 <= result[key] <= 1.0
    lowest = evaluate_mfgp_coverage_from_batch(mfgp, holdout, level=0, seed=999)
    np.testing.assert_allclose(lowest["mu"], mfgp.predict(holdout.theta, fidelity=0)[0])


def test_evaluate_coverage_from_batch_rejects_event_only() -> None:
    gen, cnp = _quick_cnp()
    mfgp = _paper_mfgp(gen, cnp, n_hf_trials=6)
    bad = for_scenario("S5", seed=0).generate(n_trials=4, n_events=16, seed=0)
    with pytest.raises(ValueError, match="holdout_batch"):
        evaluate_mfgp_coverage_from_batch(mfgp, bad)


def test_synthetic_wrapper_parity_with_batches() -> None:
    """synthetic_levels ≡ levels_from_batches on the same draws and split seeds."""
    gen, cnp = _quick_cnp()
    seed = 7
    # MC sampling in cnp_trial_predictive consumes torch's global RNG: anchor both runs.
    torch.manual_seed(42)
    X_wrap, Y_wrap = synthetic_levels(
        cnp,
        gen,
        PAPER_LEVELS,
        n_trials={"lf": 30, "hf": 6},
        n_events={"lf": 32, "hf": 64},
        n_mc_samples=8,
        seed=seed,
    )
    torch.manual_seed(42)
    lf = gen.generate(n_trials=30, n_events=32, seed=seed)
    hf = gen.generate(n_trials=6, n_events=64, seed=seed + 100)
    X_batch, Y_batch = levels_from_batches(
        cnp,
        [
            BatchLevel("model_mean", lf, 16, seed + 1),
            BatchLevel("model_mean", hf, 32, seed + 101),
            BatchLevel("observed_fraction", hf, 32, seed + 101),
        ],
        n_mc_samples=8,
    )
    for a, b in zip(X_wrap, X_batch, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(Y_wrap, Y_batch, strict=True):
        np.testing.assert_allclose(a, b)
