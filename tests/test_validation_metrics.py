"""Validation metrics for event probabilities and the fixed evaluation layout."""

import json

import numpy as np
import pytest
from scipy.special import logit

from core.surrogate_cnp import split_context_target
from core.surrogates.base import EventPrediction
from core.surrogates.evaluation import evaluate_surrogate, probability_metrics
from core.surrogates.readout import (
    bootstrap_intervals,
    context_sweep,
    spearman_noise_ceiling,
)
from data.pseudo_generator import for_scenario
from schemas.data_models import InputMode, StandardBatch
from schemas.surrogates import SurrogateRunConfig


def drawn(rates, n_events, seed=0):
    """Voxels with constant true rate per voxel: (batch, true probabilities [B, N])."""
    rng = np.random.default_rng(seed)
    p = np.repeat(np.asarray(rates, dtype=float)[:, None], n_events, axis=1)
    labels = (rng.random(p.shape) < p).astype(np.int64)
    theta = np.arange(len(rates), dtype=float)[:, None]
    batch = StandardBatch(mode=InputMode.DESIGN_ONLY, theta=theta, labels=labels)
    return batch, p


class Fixed:
    """Context-free model returning given probabilities [B, N]."""

    uses_context = False

    def __init__(self, probabilities):
        self.logits = logit(probabilities)

    def predict(self, target, context=None):
        return EventPrediction(self.logits, event_counts=target.event_counts)


def test_true_probabilities_pass_every_calibration_check():
    rng = np.random.default_rng(1)
    batch, p = drawn(rng.uniform(0.002, 0.05, 2000), 400)
    metrics, arrays = probability_metrics(batch, batch.labels, p)
    assert abs(metrics["voxel_z_mean"]) < 0.1
    assert abs(metrics["voxel_z2_mean"] - 1) < 0.15
    assert abs(metrics["calibration_z"]) < 3
    assert abs(metrics["voxel_rate_mse_denoised"]) < 1e-5  # noise sd ~2e-6
    assert metrics["calibration_chi2_per_bin"] < 3
    assert arrays["calibration_events"].sum() == batch.labels.size
    # Twice the true rate: too many predicted hits, positive real error.
    high, _ = probability_metrics(batch, batch.labels, 2 * p)
    assert high["calibration_z"] < -10 and high["voxel_z_mean"] < -1
    assert high["voxel_rate_mse_denoised"] > 1e-4


def test_denoised_squared_error_is_unbiased_where_mae_prefers_zero():
    # Dim voxels: 0.5 expected hits each, the MAE example from the metric discussion.
    n = 1125
    batch, p = drawn(np.full(20000, 0.5 / n), n, seed=2)
    observed = batch.labels.mean(axis=1)
    truth = probability_metrics(batch, batch.labels, p)[0]["voxel_rate_mse_denoised"]
    low = probability_metrics(batch, batch.labels, p / 4)[0]["voxel_rate_mse_denoised"]
    assert abs(truth) < 0.05 * (0.5 / n) ** 2
    assert low == pytest.approx((0.75 * 0.5 / n) ** 2, rel=0.1)
    assert np.abs(observed).mean() < np.abs(observed - 0.5 / n).mean()  # MAE favours 0


def test_log_loss_skill_and_lift():
    batch, p = drawn(np.r_[np.full(50, 0.01), np.full(50, 0.1)], 500, seed=3)
    constant = np.full_like(p, batch.labels.mean())
    flat, _ = evaluate_surrogate(Fixed(constant), batch, reference_rate=batch.labels.mean())
    sharp, _ = evaluate_surrogate(Fixed(p), batch, reference_rate=batch.labels.mean())
    assert flat["log_loss_skill"] == pytest.approx(0, abs=1e-12)
    assert sharp["log_loss_skill"] > 0.05
    assert sharp["lift"] == pytest.approx(sharp["average_precision"] / sharp["prevalence"])
    assert evaluate_surrogate(Fixed(p), batch)[0]["log_loss_skill"] is None


def test_fixed_layout_shares_targets_and_nests_contexts():
    batch = for_scenario("S1", seed=0).generate(n_trials=5, n_events=40)
    splits = {n: split_context_target(batch, n, seed=7, n_pool=12) for n in (3, 8, 12)}
    for n, (context, target) in splits.items():
        assert target.n_events == 28 and context.n_events == n
        np.testing.assert_array_equal(target.phi, splits[12][1].phi)
        np.testing.assert_array_equal(context.phi, splits[12][0].phi[:, :n])
    with pytest.raises(ValueError, match="n_pool"):
        split_context_target(batch, 8, seed=7, n_pool=4)
    # Without n_pool the old split is unchanged.
    old = split_context_target(batch, 8, seed=7)
    assert old[1].n_events == 32


def test_run_config_context_pool():
    base = dict(
        model={"kind": "cnp", "encoder": {"type": "mlp", "latent_dim": 8, "hidden_dims": [8]}},
        training={
            "backend": "neural",
            "n_steps": 2,
            "eval_every": 1,
            "batch_size": 2,
            "n_events": 64,
        },
        data_directory="d",
        output_directory="o",
        validation_context_events=200,
    )
    assert SurrogateRunConfig(**base).validation_context_pool == 200
    config = SurrogateRunConfig(**base, validation_context_sweep=[16, 375])
    assert config.validation_context_pool == 375
    with pytest.raises(ValueError, match="unique positive"):
        SurrogateRunConfig(**base, validation_context_sweep=[16, 16])


def test_noise_ceiling_and_bootstrap_are_seeded():
    rng = np.random.default_rng(4)
    batch, p = drawn(rng.uniform(0.005, 0.05, 200), 300, seed=4)
    ceiling = spearman_noise_ceiling(batch, p, seed=1, draws=50)
    assert 0.5 < ceiling < 1
    assert ceiling == spearman_noise_ceiling(batch, p, seed=1, draws=50)
    assert spearman_noise_ceiling(batch, np.full_like(p, 0.01), seed=1, draws=5) is None
    intervals = bootstrap_intervals(batch, batch.labels, logit(p), seed=1, draws=50)
    point, _ = evaluate_surrogate(Fixed(p), batch)
    for key in ("bernoulli_log_loss", "average_precision", "spearman_r"):
        low, high = intervals[f"{key}_ci95"]
        assert low <= point[key] <= high
    assert intervals == bootstrap_intervals(batch, batch.labels, logit(p), seed=1, draws=50)


class ContextRate:
    """Predicts each voxel's context hit fraction (plus a floor) for all targets."""

    uses_context = True

    def predict(self, target, context=None):
        rate = context.labels.mean(axis=1, keepdims=True) + 0.01  # [B, 1]
        return EventPrediction(np.broadcast_to(logit(rate), target.labels.shape).copy())


def test_context_sweep_rows_and_zeroed_labels():
    rng = np.random.default_rng(5)
    batch, _ = drawn(rng.uniform(0.05, 0.3, 60), 200, seed=5)
    config = SurrogateRunConfig(
        model={"kind": "cnp", "encoder": {"type": "mlp", "latent_dim": 8, "hidden_dims": [8]}},
        training={
            "backend": "neural",
            "n_steps": 2,
            "eval_every": 1,
            "batch_size": 2,
            "n_events": 64,
        },
        data_directory="d",
        output_directory="o",
        validation_context_events=50,
        validation_context_sweep=[10, 100],
    )
    rows = context_sweep(ContextRate(), batch, config)
    assert [(r["n_context"], r["context_labels"]) for r in rows] == [
        (10, "observed"),
        (50, "observed"),
        (50, "zeroed"),
        (100, "observed"),
    ]
    by = {(r["n_context"], r["context_labels"]): r for r in rows}
    # The model uses its context labels: zeroing them breaks the voxel trend.
    assert by[50, "observed"]["spearman_r"] > 0.5
    assert by[50, "zeroed"]["spearman_r"] is None
    assert by[100, "observed"]["spearman_r"] > by[10, "observed"]["spearman_r"]


def test_experiment_writes_readout_and_sweep(tmp_path):
    from core.surrogates.experiment import run_experiment

    source = for_scenario("S1", seed=1)
    root = tmp_path / "data"
    for split, fidelity, seed in [("train", "lf", 1), ("validation", "lf", 2)]:
        batch = source.generate(n_trials=4, n_events=24, seed=seed)
        batch.labels[:, 0], batch.labels[:, 1] = 1, 0
        folder = root / "batches" / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{fidelity}.npz",
            mode=batch.mode.value,
            theta=batch.theta,
            phi=batch.phi,
            labels=batch.labels,
        )
    config = SurrogateRunConfig(
        model={"kind": "cnp", "encoder": {"type": "mlp", "latent_dim": 8, "hidden_dims": [8]}},
        training={
            "backend": "neural",
            "n_steps": 4,
            "eval_every": 2,
            "batch_size": 2,
            "n_events": 12,
            "context": {"size": "fixed", "n": 3},
        },
        selection="bernoulli_log_loss",
        hf_validation=False,
        data_directory=root,
        output_directory=tmp_path / "run",
        validation_context_events=4,
        validation_context_sweep=[2, 8],
    )
    output = run_experiment(config)
    rows = json.loads((output / "metrics.json").read_text())
    validation = [r for r in rows if r["split"] == "validation"]
    assert all("spearman_noise_ceiling" in r and "log_loss_skill" in r for r in validation)
    assert all("spearman_noise_ceiling" not in r for r in rows if r["split"] == "train")
    assert all(r["voxel_z2_mean"] is not None for r in rows)
    sweep = json.loads((output / "context_sweep.json").read_text())
    assert {r["n_context"] for r in sweep} == {2, 4, 8}
    assert len(sweep) == 2 * 4  # best + final, three sizes + zeroed
    history = json.loads((output / "history.json").read_text())
    assert all("log_loss_skill" in r for r in history if "bernoulli_log_loss" in r)
    assert (output / "lf_calibration.png").exists()
    assert (output / "context_sweep.png").exists()
