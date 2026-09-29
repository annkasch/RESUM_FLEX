"""Real-event target oversampling with per-voxel sampling corrections."""
import numpy as np
import torch
from torch.nn import functional as F
from schemas.data_models import StandardBatch


class RealTargetSampler:
    """Uniform real contexts; uniform or stratified real targets from their complement.

    Voxels have equal sampling probability. With positive_fraction set, both
    classes are sampled uniformly within their remaining event pools. Weights
    p/q and (1-p)/(1-q) use the remaining pool's empirical prevalence and actual
    integer allocation, independently for each voxel. This estimates the same
    conditional empirical risk as uniform targets. Single-class pools retain
    their real labels and unit weights. Repeated targets are allowed when a
    class has too few distinct events; contexts and targets never overlap.
    Separate context/target RNGs match contexts across sampling strategies.
    """

    def __init__(self, batch, *, seed, batch_size, n_events, n_context_min,
                 n_context_max, positive_fraction=None):
        if positive_fraction is not None and not 0 < positive_fraction < 1:
            raise ValueError('positive_fraction must be strictly between zero and one')
        if not 1 <= n_context_min <= n_context_max <= n_events - 2:
            raise ValueError('Context range must leave at least two target events')
        if batch_size < 1 or n_events > batch.n_events:
            raise ValueError('Invalid batch or episode size')
        self.batch, self.batch_size, self.n_events = batch, batch_size, n_events
        self.n_context_min, self.n_context_max = n_context_min, n_context_max
        self.positive_fraction = positive_fraction
        self.context_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self.target_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.last_provenance = []

    def next(self):
        b, crng, trng = self.batch, self.context_rng, self.target_rng
        trials = crng.choice(b.batch_size, self.batch_size, replace=True)
        nc = int(crng.integers(self.n_context_min, self.n_context_max + 1))
        nt = self.n_events - nc
        contexts, targets, weights = [], [], []
        self.last_provenance = []
        for trial in trials:
            context = crng.choice(b.n_events, nc, replace=False)
            mask = np.ones(b.n_events, dtype=bool)
            mask[context] = False
            pool = np.flatnonzero(mask)
            positive = pool[b.labels[trial, pool] == 1]
            negative = pool[b.labels[trial, pool] == 0]
            p = len(positive) / len(pool)
            if self.positive_fraction is None or not len(positive) or not len(negative):
                target = trng.choice(pool, nt, replace=False)
                weight = np.ones(nt)
            else:
                npos = min(nt - 1, max(1, round(nt * self.positive_fraction)))
                nneg = nt - npos
                target = np.concatenate([
                    trng.choice(positive, npos, replace=npos > len(positive)),
                    trng.choice(negative, nneg, replace=nneg > len(negative))])
                q = npos / nt
                weight = np.concatenate([np.full(npos, p / q),
                                         np.full(nneg, (1-p) / (1-q))])
                permutation = trng.permutation(nt)
                target, weight = target[permutation], weight[permutation]
            contexts.append(context); targets.append(target); weights.append(weight)
            self.last_provenance.append(dict(trial=int(trial), context=context,
                target=target, pool_positive_fraction=p))
        def gather(indices):
            indices = np.asarray(indices)
            return StandardBatch(mode=b.mode,
                theta=None if b.theta is None else b.theta[trials],
                phi=None if b.phi is None else b.phi[trials[:, None], indices],
                labels=b.labels[trials[:, None], indices])
        return gather(contexts), gather(targets), np.asarray(weights)


def sampling_corrected_bce(probability, labels, weights):
    """Average importance-weighted BCE, without rebalancing the true prevalence."""
    labels = torch.as_tensor(labels, dtype=probability.dtype, device=probability.device)
    weights = torch.as_tensor(weights, dtype=probability.dtype, device=probability.device)
    losses = F.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), labels, reduction='none')
    return (weights * losses).mean()
