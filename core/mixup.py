"""Within-voxel permutation and class-aware mixup augmentation."""

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
    """Fresh, label-independent source split for each voxel in every batch.

    Voxels are sampled uniformly with replacement. Their full event sets are
    randomly partitioned in the nominal context/target proportion on each draw.
    Each side mixes negative anchors with positive partners from its own pool,
    using independent Beta weights. Single-class pools emit real events.
    A singleton positive can therefore change sides between batches.

    Real contexts are uniform draws from the context pool. Positive partners
    may be reused. Real events and negative anchors use replacement only when
    needed to fill the requested output size.
    No once-per-epoch coverage guarantee is made. With equal seeds, switching
    mix_context leaves target batches and source splits identical.
    """

    def __init__(self, batch, *, alpha, seed, batch_size, n_events, n_context, mix_context=True):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Class-aware mixup requires positive finite alpha")
        if not 0 < n_context < n_events or batch_size < 1:
            raise ValueError("Invalid mixup episode dimensions")
        if batch.n_events < 2 or batch.batch_size < 1:
            raise ValueError("Class-aware mixup requires at least two source events per voxel")
        self.batch, self.alpha = batch, alpha
        self.mix_context = mix_context
        self.context_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.nc, self.nt = n_context, n_events - n_context
        self.context_pool_size = min(
            batch.n_events - 1, max(1, round(batch.n_events * n_context / n_events))
        )
        self.last_provenance = []

    @staticmethod
    def _draw(rng, pool, size):
        return rng.choice(pool, size=size, replace=len(pool) < size)

    def next(self):
        trials = self.rng.integers(self.batch.batch_size, size=self.batch_size)
        pools = []
        for _ in trials:
            indices = self.rng.permutation(self.batch.n_events)
            pools.append((indices[:self.context_pool_size], indices[self.context_pool_size:]))

        result = []
        self.last_provenance = []
        for side, size in ((0, self.nc), (1, self.nt)):
            labels, features = [], []
            for row, voxel in enumerate(trials):
                pool = pools[row][side]
                source_labels = self.batch.labels[voxel]
                negatives = pool[source_labels[pool] == 0]
                positives = pool[source_labels[pool] == 1]
                if len(negatives) and len(positives):
                    anchors = self._draw(self.rng, negatives, size)
                    partners = self.rng.choice(positives, size=size, replace=True)
                    weight = self.rng.beta(self.alpha, self.alpha, size=size)
                else:
                    anchors = self._draw(self.rng, pool, size)
                    partners = anchors.copy()
                    weight = np.zeros(size)
                y = (1 - weight) * source_labels[anchors] + weight * source_labels[partners]
                provenance = dict(
                    row=row, voxel=int(voxel), side=side, source_pool=pool.copy(),
                    anchors=anchors.copy(), partners=partners.copy(), weights=weight.copy(),
                )
                # Consume identical augmentation draws for both context modes.
                if side == 0 and not self.mix_context:
                    real_indices = self._draw(self.context_rng, pool, size)
                    y = source_labels[real_indices]
                    provenance["real_indices"] = real_indices.copy()
                labels.append(y)
                if self.batch.phi is not None:
                    if side == 0 and not self.mix_context:
                        features.append(self.batch.phi[voxel, real_indices])
                    else:
                        features.append(
                            (1 - weight[:, None]) * self.batch.phi[voxel, anchors]
                            + weight[:, None] * self.batch.phi[voxel, partners]
                        )
                self.last_provenance.append(provenance)
            carrier = StandardBatch if side == 0 and not self.mix_context else SoftTargetBatch
            result.append(carrier(
                mode=self.batch.mode,
                theta=None if self.batch.theta is None else self.batch.theta[trials],
                phi=None if self.batch.phi is None else np.stack(features),
                labels=np.stack(labels),
            ))
        return tuple(result)
