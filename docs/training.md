# Training

Loss, sampling, mixup and episode settings for the event models.

## Loss, sampling and weights

These settings are deliberately separate:

- `focal_gamma: 0` is stable BCE on logits; positive gamma enables focal BCE.
- Natural sampling has no quota. It can use no weights or explicit class weights.
- `positive_quota` requires `sampling_correction`, using the existing per-voxel
  sampler and actual integer quota. No extra class rebalancing is added.
- BDT fits all original events, optionally with explicit class weights. It rejects
  neural losses, oversampling settings and sampling correction.

The API rejects unknown keys, including legacy top-level mixup_alpha and
Gaussian-loss settings. Class-aware mixup uses the nested sampling config described below.
There is no silent fallback from one loss or sampler to another. Class weights
and focal losses can change probability calibration; the saved config records
these choices. AP is explicitly undefined (`None`, with empty PR arrays) for
validation sets with no positives; AP selection then fails before fitting.

## Class-aware mixup

`core.mixup.ClassAwareMixupSource` samples voxels uniformly with replacement.
For each voxel occurrence in each batch, it freshly partitions the full source
set into disjoint context and target pools, independently of labels, in the
requested context/target proportion (context-free models: all events are
target pool). Both parents of a mixture come from the
same pool. Positive partners can repeat within their pool; weights are independent
per pair. A pool with only one class returns real events. A lone positive may
serve either side across batches; it is not permanently reserved for targets.

`mix_context=False` samples real contexts; `True` mixes context too. For equal
seeds, both modes use identical splits and targets. Real-event and negative-anchor
sampling uses replacement only if a pool is too small for the requested batch.
This is batch sampling, without a once-per-epoch negative coverage guarantee.
CNP, MLP, and transformer training support this sampler through the shared API:

```yaml
training:
  backend: neural
  sampling:
    strategy: class_aware_mixup
    mixup:
      alpha: 0.2
      mix_context: false
  weighting:
    strategy: none
```

Run `python -m core.surrogates train CONFIG.yaml` with these settings.
Change the model kind for MLP or transformer. Only CNP uses context observations;
other neural models use the same mixed targets. The context size follows `training.context`
(see "Episode building" below). Mixup defaults to real context.
Stable BCE (`focal_gamma: 0`) accepts soft target labels. BDT does not support
this sampler. Sampling correction and class weights are rejected with mixup;
no probability-bias correction is implied. Positive quotas cannot be combined
with it. Permutation mixup remains a standalone utility.

### Episode building

Each optimizer step draws `batch_size` voxels (with replacement) and `n_events`
events from each voxel. Context-free models (mlp, transformer, field) score all of them;
`context`, `targets` and `mixup.mix_context` apply to context models (cnp, legacy_cnp)
only and are rejected otherwise. For context models the events are split into context
and targets:

```yaml
training:
  batch_size: 16        # voxels per step
  n_events: 750         # events per voxel per step
  targets: disjoint     # or joint
  context:
    size: random        # one size per step, uniform in [n_min, cap], shared by all voxels
    n_min: 16
    max_fraction: 0.5   # cap = floor(0.5 * n_events) = 375; null = no cap
# fixed size instead:
#  context: {size: fixed, n: 250}
```

- `targets: disjoint` (default) scores only the non-context events. `joint` also
  scores the context events, as in Garnelo et al. (2018); it is limited to
  natural sampling. With the same seed both modes draw the same events. Joint
  puts loss weight on labels the model has already seen in its context, so it
  can over-trust the context; evaluation is always disjoint.
- Without a cap, the context may use up to `n_events - 2` (disjoint) or
  `n_events - 1` (joint) events.
- Old configs with `n_context_min` / `n_context_max` still load. They become a
  random context with an absolute cap `context.n_max`, so their range does not
  change with `n_events` (e.g. in fine-tuning).
- Run length: `n_steps`, or `n_passes` (not both; default 10000 steps). One pass
  means as many scored target events as there are training events. The steps are
  `ceil(n_passes * training_events / targets_per_step)`, with
  `targets_per_step = batch_size * (n_events - mean context size)` for disjoint and
  `batch_size * n_events` for joint targets and context-free models; the mean context size of a random
  context is `(n_min + cap) / 2`. Example: 1999 voxels x 1500 events, 16 x 750,
  context 16-375, `n_passes: 10` -> 3380 steps.
- The sampling audit records `targets` (`all_events` for context-free models), the
  resolved `context_range` (`[0, 0]` without context), `n_steps`,
  `requested_passes` and the realized `target_passes`.

Validation and reported train/validation metrics use real binary events. The
sampling audit records soft-label mass, mean target label, and soft-label event
count instead of treating soft labels as positive-event counts. Resolved mixup
settings are saved in checkpoints and experiment metadata.

## Optional real-plus-mixup BCE

For CNP, MLP, or transformer, add this to class-aware mixup training:

```yaml
training:
  sampling:
    strategy: class_aware_mixup
    mixup:
      alpha: 0.2
      mix_context: false
  weighting:
    strategy: none
  focal_gamma: 0
  objective:
    strategy: real_plus_mixup
    mixup_loss_weight: 0.001
    real_target_ratio: 1.0
```

The objective is `mean(real BCE) + mixup_loss_weight * mean(mixed BCE)`.
Both branches use the same context observations and model parameters, with one
optimizer update. Real targets are sampled uniformly from that batch's target
source pool, independently of labels. No source event crosses the context/target
split. Positive source events can be reused as mixture parents. Real targets
are sampled without replacement unless the requested count exceeds the pool.

`real_target_ratio` adds `max(1, round(ratio * mixed_target_count))` real targets
per voxel; it does not reduce the mixed target count. A ratio of 1 doubles target
predictions. Each branch is averaged separately, so its sample count does not
implicitly multiply its loss weight. Zero mixup loss weight is allowed. The
Beta `alpha` and the loss coefficient have distinct meanings. Extra class weights
and focal loss are rejected for this objective. `objective.strategy: single`
remains the default and uses the existing one-branch objective.

The sampler's optional `next(real_target_ratio=...)` returns
`(context, mixed_targets, real_targets)`; without the argument its two-result
interface and mixed-batch random sequence are unchanged. Shared training handles
this interface automatically.

History records `training_real_bce`, `training_mixup_bce`, and their weighted sum
as `training_loss`, averaged over each reporting interval. Checkpoints record
objective settings and sampling counts; the audit includes both target counts,
real positive counts and total target predictions. Validation still uses real
binary events. This is an experimental augmentation objective, not an unbiased
probability correction or an established performance improvement.

To try this objective, set `training.objective` in a training config as above.
