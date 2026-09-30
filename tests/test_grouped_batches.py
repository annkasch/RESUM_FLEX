"""Unequal event counts preserve voxel sampling and per-voxel denominators."""
import numpy as np
import pytest
from scipy.special import logit

from core.surrogates.base import EventPrediction
from core.surrogates.pipeline import prepare_surrogate_datasets
from core.surrogates.evaluation import evaluate_surrogate
from data.grouped_batches import GroupedEpisodeSampler
from schemas.data_models import StandardBatch, InputMode
from schemas.surrogates import NeuralTraining


def batch(n_voxels, n_events, offset):
    theta = np.arange(offset, offset + n_voxels, dtype=float)[:, None]
    labels = np.zeros((n_voxels, n_events))
    labels[:, ::3] = 1
    phi = np.broadcast_to(np.arange(n_events)[None, :, None], (n_voxels, n_events, 1)).copy()
    return StandardBatch(mode=InputMode.FULL, theta=theta, phi=phi, labels=labels)


@pytest.mark.parametrize('strategy', ['natural', 'class_aware_mixup'])
def test_uniform_voxels_and_reproducibility(strategy):
    groups = [batch(1, 20, 0), batch(3, 40, 1)]
    cfg = NeuralTraining(n_steps=1, batch_size=16, n_events=12,
                         n_context_min=4, n_context_max=4,
                         sampling={'strategy': strategy, **({'mixup': {'alpha': .2, 'mix_context': True}}
                                                          if strategy == 'class_aware_mixup' else {})})
    a, b = GroupedEpisodeSampler(groups, cfg), GroupedEpisodeSampler(groups, cfg)
    counts = np.zeros(4, dtype=int)
    for _ in range(100):
        first, second = a.next(), b.next()
        c, t = first[:2]
        np.testing.assert_array_equal(c.theta, t.theta)
        np.testing.assert_array_equal(t.labels, second[1].labels)
        assert c.labels.shape == (16, 4) and t.labels.shape == (16, 8)
        counts += np.bincount(c.theta[:, 0].astype(int), minlength=4)
        if strategy == 'natural':
            for context, target in zip(c.phi, t.phi):
                assert not set(context.ravel()) & set(target.ravel())
    assert np.max(np.abs(counts / counts.sum() - .25)) < .04


class ConstantPerVoxel:
    def predict(self, target, context=None):
        p = np.broadcast_to(.2 + .1 * target.theta, target.labels.shape)
        return EventPrediction(logit(p))


def test_gp_means_and_grouped_metrics_use_actual_counts():
    groups = [batch(1, 20, 0), batch(3, 40, 1)]
    groups[0].labels[:] = 1
    model = ConstantPerVoxel()
    data = prepare_surrogate_datasets(model, groups, groups[0], n_lf_context=4, n_hf_context=4)
    np.testing.assert_allclose(data['Y_lf_cnp'].ravel(), [.2, .3, .4, .5])
    assert data['X_lf'].shape == (4, 1)
    metrics, arrays = evaluate_surrogate(model, groups, context=groups)
    assert metrics['events'] == 140
    assert arrays['voxel_event_counts'].tolist() == [20, 40, 40, 40]
    assert metrics['mean_predicted'] == pytest.approx(.35)
    assert metrics['mean_observed'] == pytest.approx(np.concatenate([g.labels.mean(1) for g in groups]).mean())


def test_event_count_levels_order_and_preserve_observations():
    from core.surrogates.pipeline import mfgp_level_arrays

    data = dict(X_lf=np.arange(6)[:, None], Y_lf_cnp=np.arange(6)[:, None] / 10,
                X_hf=np.array([[9.]]), Y_hf_cnp=np.array([[.4]]), Y_hf_raw=np.array([[.01]]))
    counts = np.array([1500, 500, 1000, 750, 500, 1500])
    xs, ys, names, mapping = mfgp_level_arrays(data, counts, lf_levels="by_event_count")
    assert len(xs) == 6
    assert mapping == {500: 0, 750: 1, 1000: 2, 1500: 3}
    assert [len(x) for x in xs] == [2, 1, 1, 2, 1, 1]
    for size, level in mapping.items():
        np.testing.assert_array_equal(xs[level], data['X_lf'][counts == size])
        np.testing.assert_array_equal(ys[level], data['Y_lf_cnp'][counts == size])
    np.testing.assert_array_equal(ys[-1], data['Y_hf_raw'])
    pooled_x, pooled_y, _, _ = mfgp_level_arrays(data, counts)
    np.testing.assert_array_equal(pooled_x[0], data['X_lf'])
    np.testing.assert_array_equal(pooled_y[0], data['Y_lf_cnp'])
