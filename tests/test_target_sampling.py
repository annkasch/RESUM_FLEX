import numpy as np
import pytest
import torch
from core.target_sampling import RealTargetSampler, sampling_corrected_bce
from schemas.data_models import StandardBatch, InputMode


def batch():
    labels = np.zeros((3, 100), dtype=int)
    labels[0, :3] = 1
    labels[2] = 1
    phi = np.broadcast_to(np.arange(100)[None, :, None], (3,100,1)).astype(float).copy()
    return StandardBatch(mode=InputMode.FULL, theta=np.arange(3)[:,None], phi=phi, labels=labels)


def test_matching_real_contexts_and_disjoint_targets():
    b = batch()
    kwargs = dict(seed=5, batch_size=8, n_events=40, n_context_min=5, n_context_max=10)
    uniform = RealTargetSampler(b, **kwargs)
    weighted = RealTargetSampler(b, positive_fraction=.25, **kwargs)
    for _ in range(10):
        cu, _, _ = uniform.next()
        cw, targets, weights = weighted.next()
        np.testing.assert_array_equal(cu.phi, cw.phi)
        np.testing.assert_array_equal(cu.labels, cw.labels)
        np.testing.assert_allclose(weights.mean(1), 1)
        for j, provenance in enumerate(weighted.last_provenance):
            trial, ci, ti = (provenance[k] for k in ('trial','context','target'))
            assert not set(ci) & set(ti)
            np.testing.assert_array_equal(targets.phi[j], b.phi[trial,ti])
            np.testing.assert_array_equal(targets.labels[j], b.labels[trial,ti])
            np.testing.assert_allclose((weights[j]*targets.labels[j]).mean(),
                                       provenance['pool_positive_fraction'])


def test_weighted_loss_and_gradient_equal_population_for_class_constant_predictions():
    p, q = .03, .25
    y = np.array([[1]*25+[0]*75])
    w = np.where(y, p/q, (1-p)/(1-q))
    logit = torch.tensor(-2., dtype=torch.float64, requires_grad=True)
    loss = sampling_corrected_bce(logit.sigmoid().expand(1,100), y, w)
    loss.backward()
    expected = -p*np.log(torch.sigmoid(logit).item())-(1-p)*np.log(1-torch.sigmoid(logit).item())
    assert loss.item() == pytest.approx(expected)
    assert logit.grad.item() == pytest.approx(torch.sigmoid(logit).item()-p)


@pytest.mark.parametrize('fraction', [0, 1, -.1, float('nan')])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        RealTargetSampler(batch(), seed=0, batch_size=2, n_events=20,
                          n_context_min=3, n_context_max=5, positive_fraction=fraction)


def test_natural_targets_are_unique_and_class_weights_only_change_loss():
    b = batch()
    sampler = RealTargetSampler(b, seed=17, batch_size=8, n_events=40,
                                n_context_min=5, n_context_max=10, positive_fraction=None)
    p = b.labels.mean()
    for _ in range(10):
        context, target, sampling_weights = sampler.next()
        np.testing.assert_array_equal(sampling_weights, np.ones_like(target.labels))
        labels_before = target.labels.copy()
        weights = np.where(target.labels == 1, .5/p, .5/(1-p))
        np.testing.assert_array_equal(target.labels, labels_before)
        for record in sampler.last_provenance:
            assert len(np.unique(record['target'])) == len(record['target'])
            assert not set(record['context']) & set(record['target'])
        assert np.all(weights[target.labels == 1] == .5/p)
        assert np.all(weights[target.labels == 0] == .5/(1-p))
