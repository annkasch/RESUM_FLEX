"""Real-event target oversampling with per-design-point sampling corrections."""

import numpy as np

from schemas.data_models import StandardBatch, valid_event_mask


class RealTargetSampler:
    """Uniform real contexts; uniform or stratified real targets from their complement.

    Design points have equal sampling probability. With positive_fraction set, both
    classes are sampled uniformly within their remaining event pools. Weights
    p/q and (1-p)/(1-q) use the remaining pool's empirical prevalence and actual
    integer allocation, independently for each design point. This estimates the same
    conditional empirical risk as uniform targets. Single-class pools retain
    their real labels and unit weights. Repeated targets are allowed when a
    class has too few distinct events; contexts and targets never overlap.
    Separate context/target RNGs match contexts across sampling strategies.
    With unequal event counts, each design point samples from its own valid events, so
    every design point has equal weight; episodes must fit the smallest design point.
    With joint=True (natural sampling only) the targets are the context followed
    by the same non-context draws, so both modes see identical events per seed.
    Context size 0 (context-free models): every drawn event is a target and the
    context is None.
    target_counts counts how often each event was drawn as a target; coverage()
    reports the share of events and of positives that were a target at least once.
    """

    def __init__(
        self,
        batch,
        *,
        seed,
        batch_size,
        n_events,
        n_context_min,
        n_context_max,
        positive_fraction=None,
        joint=False,
    ):
        if positive_fraction is not None and not 0 < positive_fraction < 1:
            raise ValueError("positive_fraction must be strictly between zero and one")
        if joint and positive_fraction is not None:
            raise ValueError("Joint targets require natural sampling")
        if not 0 <= n_context_min <= n_context_max <= n_events - (1 if joint else 2):
            raise ValueError("Context range must leave enough target events")
        if batch_size < 1 or n_events > int(batch.events_per_design_point.min()):
            raise ValueError("Invalid batch or episode size")
        self.batch, self.batch_size, self.n_events = batch, batch_size, n_events
        self.n_context_min, self.n_context_max = n_context_min, n_context_max
        self.positive_fraction = positive_fraction
        self.joint = joint
        self.context_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self.target_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.last_provenance = []
        self.target_counts = np.zeros(batch.labels.shape, dtype=np.int32)  # [B, N]

    def next(self):
        b, crng, trng = self.batch, self.context_rng, self.target_rng
        trials = crng.choice(b.batch_size, self.batch_size, replace=True)
        nc = int(crng.integers(self.n_context_min, self.n_context_max + 1))
        nt = self.n_events - nc
        contexts, targets, weights = [], [], []
        self.last_provenance = []
        counts = b.events_per_design_point
        for trial in trials:
            n = int(counts[trial])
            context = crng.choice(n, nc, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[context] = False
            pool = np.flatnonzero(mask)
            positive = pool[b.labels[trial, pool] == 1]
            negative = pool[b.labels[trial, pool] == 0]
            p = len(positive) / len(pool)
            if self.positive_fraction is None or not len(positive) or not len(negative):
                target = trng.choice(pool, nt, replace=False)
                if self.joint:
                    target = np.concatenate([context, target])
                weight = np.ones(len(target))
            else:
                npos = min(nt - 1, max(1, round(nt * self.positive_fraction)))
                nneg = nt - npos
                target = np.concatenate(
                    [
                        trng.choice(positive, npos, replace=npos > len(positive)),
                        trng.choice(negative, nneg, replace=nneg > len(negative)),
                    ]
                )
                q = npos / nt
                weight = np.concatenate([np.full(npos, p / q), np.full(nneg, (1 - p) / (1 - q))])
                permutation = trng.permutation(nt)
                target, weight = target[permutation], weight[permutation]
            np.add.at(self.target_counts[trial], target, 1)
            contexts.append(context)
            targets.append(target)
            weights.append(weight)
            self.last_provenance.append(
                dict(trial=int(trial), context=context, target=target, pool_positive_fraction=p)
            )

        def gather(indices):
            indices = np.asarray(indices)
            return StandardBatch(
                mode=b.mode,
                theta=None if b.theta is None else b.theta[trials],
                phi=None if b.phi is None else b.phi[trials[:, None], indices],
                labels=b.labels[trials[:, None], indices],
            )

        context = gather(contexts) if nc else None
        return context, gather(targets), np.asarray(weights)

    def coverage(self) -> dict:
        """Share of valid events (and positives) drawn as a target at least once."""
        b = self.batch
        valid = valid_event_mask(b.event_counts, b.labels.shape)  # [B, N]
        counts = self.target_counts[valid]
        positive = b.labels[valid] == 1
        seen = counts > 0
        return dict(
            target_event_coverage=float(seen.mean()),
            target_positive_coverage=float(seen[positive].mean()) if positive.any() else None,
            target_count_median=float(np.median(counts)),
            target_count_max=int(counts.max()),
        )
