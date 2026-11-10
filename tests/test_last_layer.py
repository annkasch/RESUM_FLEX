"""Bayesian (Laplace) last layer: math, fit, draws, persistence and pipeline outputs."""

import json

import numpy as np
import pytest
import torch

from core.gp_metrics import count_predictive_scores
from core.last_layer import laplace_posterior, log_evidence
from core.surrogate_cnp import split_context_target
from core.surrogates import load_surrogate
from core.surrogates.datasets import NpzBatches
from core.surrogates.last_layer import (
    design_point_probability_draws,
    feature_chunks,
    fit_last_layer,
    last_layer_weights,
)
from core.surrogates.models import build_surrogate
from data.pseudo_generator import for_scenario
from schemas.data_models import StandardBatch


def _spec(kind, **last_layer):
    settings = {"draws": 200, **last_layer}
    if kind == "cnp":
        return {
            "kind": "cnp",
            "encoder": {"type": "mlp", "latent_dim": 6, "hidden_dims": [8]},
            "last_layer": settings,
        }
    return {"kind": kind, "architecture": {"hidden_dims": [8]}, "last_layer": settings}


def _batch(seed=0, n_trials=5, n_events=40):
    return for_scenario("S1", seed=seed).generate(n_trials=n_trials, n_events=n_events, seed=seed)


def _episode(model, batch):
    if model.uses_context:
        return split_context_target(batch, 8, seed=1)
    return None, batch


def test_hessian_matches_autograd():
    """The accumulated precision is the exact Bernoulli Hessian of the last layer."""
    model = build_surrogate(_spec("mlp", prior_precision=1.0), 1, 1, seed=3)
    batch = _batch()
    posterior = fit_last_layer(model, batch)
    features = np.concatenate([f[v] for *_, f, v in feature_chunks(model, batch)])
    f = torch.as_tensor(features)
    y = torch.as_tensor(batch.labels[batch.event_mask], dtype=torch.float64)

    def nll(w):
        return torch.nn.functional.binary_cross_entropy_with_logits(f @ w, y, reduction="sum")

    w = torch.as_tensor(last_layer_weights(model))
    hessian = torch.autograd.functional.hessian(nll, w).numpy()
    precision = np.linalg.inv(posterior.covariance)
    np.testing.assert_allclose(precision - np.eye(len(w)), hessian, rtol=1e-6, atol=1e-8)
    assert posterior.events == batch.event_mask.sum()


def test_evidence_maximum_is_selected():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 4))
    hessian, mean = a.T @ a, rng.normal(size=4)
    posterior = laplace_posterior(hessian, mean, events=30)
    eigenvalues = np.linalg.eigvalsh(hessian)
    best = posterior.log_evidence
    for factor in (0.5, 2.0):
        assert log_evidence(posterior.prior_precision * factor, eigenvalues, mean) < best
    fixed = laplace_posterior(hessian, mean, prior_precision=3.0, events=30)
    assert fixed.prior_precision == 3.0


@pytest.mark.parametrize("kind", ["mlp", "cnp", "field"])
def test_draws_center_on_map_and_shrink_with_prior(kind, tmp_path):
    model = build_surrogate(_spec(kind), 1, 1, seed=0)
    context, target = _episode(model, _batch())
    fit_last_layer(model, target, context)
    draws = design_point_probability_draws(model, target, context)  # [S, B]
    assert draws.shape == (200, target.batch_size)
    assert ((draws > 0) & (draws < 1)).all()
    mapped = model.predict(target, context=context).mean
    # Wide uncertainty moves the mean of sigmoid draws only slightly at small logit sd.
    assert np.abs(draws.mean(axis=0) - mapped).max() < 0.1
    # A very strong prior collapses the draws onto the trained weights.
    tight = build_surrogate(_spec(kind, prior_precision=1e12), 1, 1, seed=0)
    tight.module.load_state_dict(model.module.state_dict())
    fit_last_layer(tight, target, context)
    collapsed = design_point_probability_draws(tight, target, context)
    np.testing.assert_allclose(collapsed.mean(axis=0), mapped, atol=1e-5)
    assert collapsed.std(axis=0).max() < 1e-5
    # Persistence keeps the posterior; draws are reproducible from the seed.
    model.save(tmp_path / "ckpt")
    restored = load_surrogate(tmp_path / "ckpt")
    assert restored.capabilities["predictive_distribution"]
    np.testing.assert_allclose(design_point_probability_draws(restored, target, context), draws)


def test_saving_without_posterior_removes_stale_file(tmp_path):
    model = build_surrogate(_spec("mlp"), 1, 1, seed=0)
    batch = _batch()
    fit_last_layer(model, batch)
    model.save(tmp_path / "ckpt")
    assert (tmp_path / "ckpt/last_layer.npz").exists()
    model.last_layer = None
    model.save(tmp_path / "ckpt")
    assert not (tmp_path / "ckpt/last_layer.npz").exists()


def test_context_model_needs_context_and_unsupported_kinds_fail():
    model = build_surrogate(_spec("cnp"), 1, 1, seed=0)
    with pytest.raises(ValueError, match="context"):
        fit_last_layer(model, _batch())
    plain = build_surrogate({"kind": "mlp", "architecture": {"hidden_dims": [8]}}, 1, 1)
    with pytest.raises(ValueError, match="no Bayesian last layer"):
        fit_last_layer(plain, _batch())
    with pytest.raises(ValueError):
        build_surrogate({"kind": "transformer", "last_layer": {}}, 1, 1)


def test_ragged_design_points_average_only_valid_events():
    model = build_surrogate(_spec("mlp", prior_precision=1e12), 1, 1, seed=0)
    batch = _batch(n_trials=3, n_events=20)
    counts = np.array([20, 7, 12])
    valid = np.arange(20) < counts[:, None]
    ragged = StandardBatch(
        mode=batch.mode,
        theta=batch.theta,
        phi=np.where(valid[..., None], batch.phi, 0.0),
        labels=np.where(valid, batch.labels, 0),
        event_counts=counts,
    )
    fit_last_layer(model, ragged)
    draws = design_point_probability_draws(model, ragged)
    np.testing.assert_allclose(draws.mean(axis=0), model.predict(ragged).mean, atol=1e-5)


def test_count_predictive_scores_perfect_probabilities():
    rng = np.random.default_rng(1)
    n = np.full(400, 2000)
    p = rng.uniform(0.002, 0.01, 400)
    k = rng.binomial(n, p)
    scores, arrays = count_predictive_scores(k, n, np.tile(p, (1000, 1)), seed=0)
    for level, nominal in (("1", 0.68), ("2", 0.95)):
        assert abs(scores["coverage"][level] - nominal) < 0.08
    assert np.isfinite(scores["predictive_nll"]) and arrays["sigma"].shape == (400,)
    with pytest.raises(ValueError):
        count_predictive_scores(k, n, np.zeros((10, 400)))


class NormalizedNpz(NpzBatches):
    """Npz batches that declare an (identity) affine normalization, as inference needs."""

    def normalization(self):
        return {
            g: {"method": "standard", "offset": [0.0], "scale": [1.0]} for g in ("theta", "phi")
        }


@pytest.mark.parametrize("kind", ["mlp", "cnp"])
def test_experiment_writes_posterior_and_predictive_scores(kind, tmp_path):
    from scipy.special import expit

    from core.surrogates.experiment import run_experiment
    from core.surrogates.inference import predict_partition, read_event_logits
    from schemas.surrogates import SurrogateRunConfig

    root = tmp_path / "data"
    for split, fidelity, seed in [
        ("train", "lf", 1),
        ("validation", "lf", 2),
        ("validation", "hf", 3),
    ]:
        batch = _batch(seed=seed, n_trials=4, n_events=30)
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
        model=_spec(kind),
        training=dict(
            backend="neural",
            n_steps=4,
            eval_every=2,
            batch_size=2,
            n_events=12,
            **({"context": {"size": "fixed", "n": 3}} if kind == "cnp" else {}),
        ),
        data_directory=root,
        data_provider="tests.test_last_layer:NormalizedNpz",
        output_directory=tmp_path / "run",
        validation_context_events=4,
        selection="bernoulli_log_loss",
    )
    output = run_experiment(config)
    for name in ("best", "final"):
        assert (output / "checkpoints" / name / "last_layer.npz").exists()
    rows = json.loads((output / "metrics.json").read_text())
    held_out = [r for r in rows if r["split"] == "validation"]
    assert held_out and all("design_point_predictive" in r for r in held_out)
    assert all("design_point_predictive" not in r for r in rows if r["split"] == "train")
    assert np.isfinite(held_out[0]["design_point_predictive"]["predictive_nll"])
    assert (output / "lf_validation_best_predictive.npz").exists()
    assert (output / "lf_validation_best_predictive_coverage.png").exists()
    manifest = json.loads((output / "checkpoints/best/model.json").read_text())
    assert manifest["metadata"]["last_layer"]["prior_precision"] > 0
    if kind == "mlp":
        path = predict_partition(output / "checkpoints/best", root, "validation", "hf")
        with np.load(path) as arrays:
            assert arrays["probability_draws"].shape == (200, 4)
            assert (arrays["probability_sd"] > 0).all()
        assert not (path.parent / "validation_hf_events").exists()  # off by default
        path = predict_partition(
            output / "checkpoints/best",
            root,
            "validation",
            "lf",
            event_logits=True,
            shard_events=40,
        )
        _, counts, logits = read_event_logits(path.parent / "validation_lf_events")
        with np.load(path) as arrays:
            np.testing.assert_array_equal(counts, arrays["event_counts"])
            means = [p.mean() for p in np.split(expit(logits), np.cumsum(counts)[:-1])]
            np.testing.assert_allclose(means, arrays["mean_probability"], rtol=1e-5)
        provenance = json.loads(path.with_suffix(".json").read_text())
        assert provenance["event_logits"] == "validation_lf_events"
