"""Opt-in, within-voxel mixup of target events; real contexts stay unchanged."""

import numpy as np

from schemas.data_models import StandardBatch


class SoftTargetBatch(StandardBatch):
    """Training-only target carrier. Normal StandardBatch labels remain binary."""

    def _check_labels(self) -> None:
        if not np.isfinite(self.labels).all() or np.any((self.labels < 0) | (self.labels > 1)):
            raise ValueError("Soft mixup targets must be finite and lie in [0, 1]")


def mixup_targets(batch: StandardBatch, *, alpha: float, rng: np.random.Generator):
    """Mix within each trial after context/target partitioning.

    Independent Beta(alpha, alpha) weights per event pair preserve the mean
    target label in expectation, not exactly per batch. Pairing stays within
    each trial. Theta is unchanged. All-negative trials
    remain negative. Convex feature combinations are augmentation, not new physics
    simulations (in particular mixed momenta need not preserve photon energy).
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Mixup alpha must be positive and finite")
    perm = np.stack([rng.permutation(batch.n_events) for _ in range(batch.batch_size)])
    weight = rng.beta(alpha, alpha, size=(batch.batch_size, batch.n_events))
    paired = np.take_along_axis(batch.labels, perm, axis=1)
    labels = weight * batch.labels + (1 - weight) * paired
    phi = None
    if batch.phi is not None:
        paired_phi = np.take_along_axis(batch.phi, perm[..., None], axis=1)
        phi = weight[..., None] * batch.phi + (1 - weight[..., None]) * paired_phi
    return SoftTargetBatch(mode=batch.mode, theta=batch.theta, phi=phi, labels=labels)


class ClassAwareMixupSource:
    """Fixed disjoint source pools; negatives used once per global epoch.

    Episodes use a fixed nominal context size. Tail episodes/batches can be
    smaller. A singleton positive is reserved for targets; positive-free pools
    emit original negatives. Positive-only voxels cannot use this sampler.

    With mix_context=False, contexts are uniform samples without replacement
    from the original disjoint context pool, including its original positives.
    Target augmentation and episode order match mix_context=True exactly.
    Contexts can repeat between episodes; the negative-epoch guarantee then
    describes the scheduling stream and target usage, not actual context usage.
    """

    def __init__(self, batch, *, alpha, seed, batch_size, n_events, n_context, mix_context=True):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Class-aware mixup requires positive finite alpha')
        if not 0 < n_context < n_events or batch_size < 1:
            raise ValueError('Invalid mixup episode dimensions')
        self.batch, self.alpha = batch, alpha
        self.mix_context = mix_context
        self.context_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.nc, self.nt = n_context, n_events - n_context
        self.pools = []
        self.epoch = 0
        self.pending = []
        self.last_provenance = []
        for i, labels in enumerate(batch.labels):
            neg = self.rng.permutation(np.flatnonzero(labels == 0))
            pos = self.rng.permutation(np.flatnonzero(labels == 1))
            if len(neg) < 2:
                raise ValueError(f'Voxel {i}: class-aware pools require at least two negatives')
            cut = min(len(neg)-1, max(1, round(len(neg)*n_context/n_events)))
            pc = min(len(pos)-1, max(1, round(len(pos)*n_context/n_events))) if len(pos)>1 else 0
            self.pools.append(((neg[:cut], pos[:pc]), (neg[cut:], pos[pc:])))

    def _start_epoch(self):
        from collections import defaultdict
        buckets = defaultdict(list)
        for i, pools in enumerate(self.pools):
            cn, tn = (self.rng.permutation(p[0]) for p in pools)
            chunks = min(len(cn), len(tn), max(int(np.ceil(len(cn)/self.nc)), int(np.ceil(len(tn)/self.nt))))
            for c, t in zip(np.array_split(cn, chunks), np.array_split(tn, chunks)):
                buckets[(len(c), len(t))].append((i, c, t))
        batches = []
        for episodes in buckets.values():
            self.rng.shuffle(episodes)
            batches.extend(episodes[k:k+self.batch_size] for k in range(0,len(episodes),self.batch_size))
        self.rng.shuffle(batches)
        self.pending = batches
        self.epoch += 1

    def next(self):
        if not self.pending:
            self._start_epoch()
        episodes = self.pending.pop()
        result = []
        self.last_provenance = []
        for side in (0,1):
            labels, features, trials = [], [], []
            for i, c, t in episodes:
                negatives = c if side == 0 else t
                positives = self.pools[i][side][1]
                partners = self.rng.choice(positives, len(negatives), replace=True) if len(positives) else negatives
                weight = self.rng.beta(self.alpha,self.alpha,len(negatives)) if len(positives) else np.zeros(len(negatives))
                y = weight  # paired negative=0 and positive=1; unmixed negatives=0
                # Retain augmentation RNG draws above so targets and episode order
                # are identical in paired mixed-context / real-context experiments.
                if side == 0 and not self.mix_context:
                    pool = np.concatenate(self.pools[i][0])
                    real_indices = self.context_rng.choice(pool, len(negatives), replace=False)
                    y = self.batch.labels[i, real_indices]
                labels.append(y)
                trials.append(i)
                if self.batch.phi is not None:
                    if side == 0 and not self.mix_context:
                        features.append(self.batch.phi[i, real_indices])
                    else:
                        features.append((1-weight[:,None])*self.batch.phi[i,negatives]+weight[:,None]*self.batch.phi[i,partners])
                provenance = dict(epoch=self.epoch,voxel=i,side=side,
                                  negatives=negatives.copy(),partners=partners.copy())
                if side == 0 and not self.mix_context:
                    provenance['real_indices'] = real_indices.copy()
                self.last_provenance.append(provenance)
            carrier = StandardBatch if side == 0 and not self.mix_context else SoftTargetBatch
            result.append(carrier(mode=self.batch.mode,
                theta=None if self.batch.theta is None else self.batch.theta[trials],
                phi=None if self.batch.phi is None else np.stack(features),labels=np.stack(labels)))
        return tuple(result)
