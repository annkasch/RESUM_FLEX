"""Last-layer noise on MFGP model-mean levels (``noise: last_layer``)."""

import json

import numpy as np
import pytest

from core.mfgp_pipeline import _noise_sum, last_layer_noise
from schemas.surrogates import MFGPRunConfig, MFGPStageConfig
from tests.test_mfgp_levels import TWO_LEVELS, legacy_data


def test_level_noise_setting():
    config = MFGPStageConfig(levels=TWO_LEVELS)
    assert config.levels[0].noise == "learned"
    level = {**TWO_LEVELS[0], "noise": "last_layer"}
    assert MFGPStageConfig(levels=[level, TWO_LEVELS[1]]).levels[0].noise == "last_layer"
    with pytest.raises(ValueError):
        MFGPStageConfig(levels=[{**TWO_LEVELS[0], "noise": "binomial"}, TWO_LEVELS[1]])
    with pytest.raises(ValueError):  # observed-fraction levels have no such setting
        MFGPStageConfig(levels=[TWO_LEVELS[0], {**TWO_LEVELS[1], "noise": "last_layer"}])


def test_last_layer_noise_in_fitted_units():
    draws = np.array([[0.1, 0.2], [0.3, 0.2], [0.2, 0.2]])  # [S, n]
    noise, floored = last_layer_noise(draws, output_transform="identity")
    np.testing.assert_allclose(noise, draws.var(axis=0))
    assert floored == 0 and noise[1] == pytest.approx(0, abs=1e-20)
    noise, floored = last_layer_noise(draws, output_transform="log")
    np.testing.assert_allclose(noise, np.log(draws).var(axis=0))
    _, floored = last_layer_noise(np.array([[0.0, 0.1], [0.1, 0.1]]), output_transform="log")
    assert floored == 1
    for bad in (np.array([[0.1, 1.2], [0.1, 0.2]]), np.array([[np.nan], [0.1]])):
        with pytest.raises(ValueError, match="probabilities"):
            last_layer_noise(bad, output_transform="identity")
    with pytest.raises(ValueError, match="shape"):
        last_layer_noise(np.array([[0.1, 0.2]]), output_transform="identity")


def test_noise_sum():
    a, b = np.array([1.0, 2.0]), np.array([0.5, 0.5])
    assert _noise_sum([None, None], [None, None]) is None
    total = _noise_sum([a, None], [b, b])
    np.testing.assert_allclose(total[0], a + b)
    np.testing.assert_allclose(total[1], b)


def _level_noise_variance(gp, level):
    names = gp.model.parameter_names_flat().tolist()
    index = [i for i, n in enumerate(names) if "variance" in n and "Gaussian_noise" in n]
    return float(gp.model.param_array[index[level]])


def test_fixed_noise_takes_over_the_learned_level_noise():
    """Known per-point noise given as fixed noise leaves little for the learned level noise."""
    pytest.importorskip("GPy")
    from core.mfgp_pipeline import fit_mfgp

    rng = np.random.default_rng(np.random.SeedSequence([0, 1]))
    X0, X1 = np.linspace(0, 1, 40)[:, None], np.linspace(0, 1, 12)[:, None]
    f = np.sin(4 * X0[:, 0])
    variance = rng.uniform(0.005, 0.03, size=40)
    Y0 = (f + rng.normal(0, np.sqrt(variance)))[:, None]
    Y1 = np.sin(4 * X1)
    np.random.seed(0)
    learned = fit_mfgp([X0, X1], [Y0, Y1], n_restarts=2)
    np.random.seed(0)
    fixed = fit_mfgp([X0, X1], [Y0, Y1], n_restarts=2, point_noise=[variance, None])
    assert _level_noise_variance(learned, 0) > 0.004
    assert _level_noise_variance(fixed, 0) < 0.3 * _level_noise_variance(learned, 0)


def _last_layer_model(root, tmp_path):
    from core.surrogates import build_surrogate
    from core.surrogates.checkpoints import save_surrogate
    from core.surrogates.datasets import NpzBatches
    from core.surrogates.last_layer import fit_last_layer

    lf, _ = NpzBatches(root).load("train", "lf")
    spec = {"kind": "mlp", "architecture": {"hidden_dims": [8]}, "last_layer": {"draws": 100}}
    model = build_surrogate(spec, lf.theta.shape[1], lf.phi.shape[-1], seed=0)
    fit_last_layer(model, lf)
    save_surrogate(tmp_path / "run/checkpoints/best", model)
    return model, lf


@pytest.mark.parametrize(
    ("transform", "observation_noise"), [("identity", "learned"), ("log", "binomial")]
)
def test_standalone_run_with_last_layer_noise(tmp_path, transform, observation_noise):
    pytest.importorskip("GPy")
    from core.surrogate_mfgp import load_mfgp
    from core.surrogates.last_layer import design_point_probability_draws
    from core.surrogates.mfgp_stage import run_mfgp_config

    root = legacy_data(tmp_path / "data")
    model, lf = _last_layer_model(root, tmp_path)
    config = MFGPRunConfig(
        levels=[{**TWO_LEVELS[0], "noise": "last_layer", "max_design_points": 5}, TWO_LEVELS[1]],
        evaluation=[{"partition": "validation", "fidelity": "hf"}],
        n_restarts=1,
        output_transform=transform,
        observation_noise=observation_noise,
        data_directory=root,
        checkpoint=tmp_path / "run/checkpoints/best",
        output_directory=tmp_path / "gp",
    )
    out = run_mfgp_config(config)
    with np.load(out / "training_arrays.npz") as arrays:
        index = arrays["index_0"]
    expected, _ = last_layer_noise(
        design_point_probability_draws(model, lf.take(index)), output_transform=transform
    )
    with np.load(out / "fit_arrays.npz") as fit:
        np.testing.assert_allclose(fit["last_layer_noise_0"], expected)
        assert "last_layer_noise_1" not in fit
        assert ("point_noise_1" in fit) == (observation_noise == "binomial")
    assert (expected > 0).all()
    gp = load_mfgp(out / "model.pkl")
    np.testing.assert_allclose(gp.model.Y_metadata["point_noise"][: len(index)], expected)
    levels = json.loads((out / "levels.json").read_text())
    assert levels[0]["last_layer_draws"] == 100
    assert levels[0]["last_layer_noise_mean"] == pytest.approx(expected.mean())
    assert "last_layer_noise_mean" not in levels[1]
    assert "last-layer variance" in json.loads((out / "model.json").read_text())["noise"]


def test_last_layer_noise_needs_a_posterior(tmp_path):
    pytest.importorskip("GPy")
    from core.surrogates import build_surrogate
    from core.surrogates.checkpoints import save_surrogate
    from core.surrogates.datasets import NpzBatches
    from core.surrogates.mfgp_stage import run_mfgp_config

    root = legacy_data(tmp_path / "data")
    lf, _ = NpzBatches(root).load("train", "lf")
    model = build_surrogate(
        {"kind": "mlp", "architecture": {"hidden_dims": [8]}},
        lf.theta.shape[1],
        lf.phi.shape[-1],
        seed=0,
    )
    save_surrogate(tmp_path / "run/checkpoints/best", model)
    config = MFGPRunConfig(
        levels=[{**TWO_LEVELS[0], "noise": "last_layer"}, TWO_LEVELS[1]],
        n_restarts=1,
        data_directory=root,
        checkpoint=tmp_path / "run/checkpoints/best",
        output_directory=tmp_path / "gp",
    )
    with pytest.raises(ValueError, match="last-layer posterior"):
        run_mfgp_config(config)


def _inference_file(path, provider, *, draws):
    theta, ids = provider.theta("train", "lf")
    batch, _ = provider.load("train", "lf")
    rng = np.random.default_rng(np.random.SeedSequence([3, 0]))
    samples = rng.uniform(0.05, 0.3, size=(20, len(ids)))  # [S, B]
    arrays = dict(
        design_point_ids=np.array(ids),
        theta=theta,
        event_counts=batch.events_per_design_point,
        mean_probability=samples.mean(axis=0),
    )
    if draws:
        arrays["probability_draws"] = samples.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    meta = dict(partition="train", fidelity="lf", checkpoint="c", step=0, data="d")
    path.with_suffix(".json").write_text(json.dumps(meta))
    return samples


@pytest.mark.parametrize("draws", [True, False])
def test_inference_level_with_last_layer_noise(tmp_path, draws):
    pytest.importorskip("GPy")
    from core.surrogates.datasets import NpzBatches
    from core.surrogates.mfgp_stage import run_mfgp_config

    root = legacy_data(tmp_path / "data")
    inference = tmp_path / "inference/train_lf.npz"
    samples = _inference_file(inference, NpzBatches(root), draws=draws)
    config = MFGPRunConfig(
        levels=[{**TWO_LEVELS[0], "noise": "last_layer", "inference": inference}, TWO_LEVELS[1]],
        n_restarts=1,
        data_directory=root,
        output_directory=tmp_path / "gp",
    )
    if not draws:
        with pytest.raises(ValueError, match="no probability_draws"):
            run_mfgp_config(config)
        return
    out = run_mfgp_config(config)
    with np.load(out / "fit_arrays.npz") as fit:
        expected = samples.astype(np.float32).astype(float).var(axis=0)
        np.testing.assert_allclose(fit["last_layer_noise_0"], expected)
