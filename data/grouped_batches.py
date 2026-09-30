"""Keep unequal event counts in separate dense batches without padding."""

import numpy as np


def batch_groups(batch):
    return batch if isinstance(batch, list) else [batch]


def concatenate_batches(batches):
    first = batches[0]
    return type(first)(
        mode=first.mode,
        theta=None if first.theta is None else np.concatenate([b.theta for b in batches]),
        phi=None if first.phi is None else np.concatenate([b.phi for b in batches]),
        labels=np.concatenate([b.labels for b in batches]),
    )


class GroupedEpisodeSampler:
    """Uniform voxel draws, then existing within-voxel sampling on full files.

    Choose storage groups in proportion to voxel count, not event count. Each
    existing sampler draws one uniform voxel from its group. All rows share the
    episode context size, so their sampled outputs concatenate without padding.
    """

    def __init__(self, groups, training):
        from core.mixup import ClassAwareMixupSource
        from core.target_sampling import RealTargetSampler

        self.rng = np.random.default_rng(np.random.SeedSequence([training.seed, 17]))
        self.batch_size = training.batch_size
        self.nc_min, self.nc_max = training.n_context_min, training.n_context_max
        self.mixup = training.sampling.strategy == "class_aware_mixup"
        counts = np.array([b.batch_size for b in groups])
        self.probabilities = counts / counts.sum()
        self.samplers = []
        for i, batch in enumerate(groups):
            args = dict(seed=int(np.random.SeedSequence([training.seed, 18, i]).generate_state(1)[0]),
                        batch_size=1, n_events=training.n_events)
            if self.mixup:
                sampler = ClassAwareMixupSource(
                    batch, **args, n_context=self.nc_min,
                    alpha=training.sampling.mixup.alpha,
                    mix_context=training.sampling.mixup.mix_context,
                )
            else:
                sampler = RealTargetSampler(
                    batch, **args, n_context_min=self.nc_min, n_context_max=self.nc_max,
                    positive_fraction=training.sampling.positive_fraction,
                )
            self.samplers.append(sampler)
        self.voxel_draws = np.zeros(len(groups), dtype=int)

    def next(self, *, n_context=None, real_target_ratio=None):
        nc = int(self.rng.integers(self.nc_min, self.nc_max + 1)) if n_context is None else n_context
        groups = self.rng.choice(len(self.samplers), self.batch_size, p=self.probabilities)
        rows = []
        for index in groups:
            sampler = self.samplers[index]
            if self.mixup:
                row = sampler.next(n_context=nc, real_target_ratio=real_target_ratio)
            else:
                sampler.n_context_min = sampler.n_context_max = nc
                row = sampler.next()
            rows.append(row)
            self.voxel_draws[index] += 1
        return tuple(
            None if rows[0][i] is None else
            np.concatenate([r[i] for r in rows]) if isinstance(rows[0][i], np.ndarray) else
            concatenate_batches([r[i] for r in rows])
            for i in range(len(rows[0]))
        )


class GroupedFixedBatchSource:
    """BatchSource adapter for uniformly sampled voxels with unequal file sizes."""

    def __init__(self, groups):
        from data.batch_source import FixedBatchSource

        self.sources = [FixedBatchSource(b, replace_events=False) for b in groups]
        self.mode = self.sources[0].mode
        self.dim_theta = self.sources[0].dim_theta
        self.dim_phi = self.sources[0].dim_phi
        counts = np.array([b.batch_size for b in groups])
        self.probabilities = counts / counts.sum()

    def generate(self, n_trials, n_events, seed):
        if n_trials < 1 or n_events < 1:
            raise ValueError("n_trials and n_events must be positive")
        if any(n_events > s.batch.n_events for s in self.sources):
            raise ValueError("n_events exceeds available events in a group")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.sources), n_trials, p=self.probabilities)
        return concatenate_batches([
            self.sources[i].generate(1, n_events, seed=int(rng.integers(2**32)))
            for i in indices
        ])
