# Training on an event stream

`python -m core.surrogates stream CONFIG.yaml`

An event model can train on a large set of ungrouped events, e.g. a homogeneous simulation
campaign, instead of on events grouped by design point (`core/surrogates/stream_training.py`,
config `StreamRunConfig`). Kinds: `mlp`, `field`, `transformer` (gradient steps on batches)
and `bdt` (an in-memory case-control sample). There is no stream CNP (below).

```yaml
model: {kind: mlp, architecture: {hidden_dims: [256, 256, 256]}, last_layer: {}}
dim_theta: 2                     # input widths of the design-point data it is evaluated on
dim_phi: 3
record_position: {phi_columns: [0, 1]}   # phi columns added to theta = record position (optional)
training: {batch_size: 65536, n_passes: 10, device: cuda, precision: bf16}
event_store: /path/to/store
event_source: my_package.loader:MyEventStream   # required: the application's reader
event_source_options: {in_memory: true}         # passed to it unchanged
output_directory: runs/stream_mlp/seed0
```

The stream format (`core/event_stream.py`): float32 records [M, W + 1] = the position
(`dim_theta` columns), then the other φ columns in φ order, then the label bits. In the
example: θ + φ[0:2], φ[2], label (W = 3). At evaluation on design-point data, the record
position is θ plus the position φ columns, so the model sees the same inputs in both cases.

- `kind: mlp`: an MLP on all W normalized record columns, any layout. `record_position`
  (old name `mlp_position`, still read) names the φ columns added to θ (none: θ alone is the
  position, all φ columns follow).
- `kind: transformer`: the tabular transformer on the same normalized columns, one token per
  column (`architecture` as for design-point training); layout as for `mlp`.
- `kind: field`: the neural field (point clouds); its `position` and `direction` settings
  define the layout, e.g. `dim_theta: 3`, `dim_phi: 6`, position φ[0:3], direction φ[3:6]
  gives records x, y, z, px, py, pz, label.
- `kind: bdt`: boosted trees on the W record columns in physical units; layout as for `mlp`.
  `training` takes other settings, see below.

The event source implements the `EventSource` protocol (`n_events`, `n_positive`,
`normalization` of the W input columns, `n_batches`, `batches`, `load`); its reading and
shuffling options are its own and go into `event_source_options`. The neural models fix their
own units from the store's train normalization (`set_stream_normalization`). Checkpoints are
selected by the Bernoulli log loss on the validation events. With `last_layer` (not bdt),
the selected checkpoints get the Laplace posterior from one streaming pass over all training
events.
The stream validation events also get the calibration per record column (`record_j`,
`stream_feature_calibration.png`; `docs/evaluation.md`; bin count `feature_calibration: {bins:
10}` and `within_point_null_draws`, as in training configs). With `residual_test:` (settings
in `docs/configs.md`) the validation events get the residual learnability test split by
event (records are independent; the total gain only) and the design-point evaluation sets
get it split by design point, as in a training run. On the validation events it uses a
seeded uniform subsample with about `max_positives` positives (20 000), so its cost stays
bounded for any `validation_events`. Several GPUs (neural kinds): `torchrun`
(DistributedDataParallel). bf16 autocast and `torch.compile` are options. Run outputs label
the stream metrics with fidelity `stream`.

The mlp and transformer checkpoints are ordinary field checkpoints (a field without
encodings, with an MLP or transformer network). `predict` evaluates them on design-point
data: θ, offsets and other φ columns are mapped into the model's stream units, so no equal
normalization is required.

## BDT: case-control sample

Boosted trees do not learn batch by batch: every new tree is fitted to the gradients of all
training events, so they must sit in memory (sklearn: ~9 W + 16 bytes per event, ~70 GB per
10⁹ events at W = 6, and a full scan per tree). One pass over the training events therefore
keeps every positive and each negative with probability q = min(1, k n₊ / n₋), and weights
the kept negatives by 1 / q, so the fitted probabilities stay calibrated. At low prevalence
almost all information is in the positives; with k ≈ 10–20 the negatives lose a few percent
of theirs, and memory grows with the positives, not with the stream.

```yaml
model: {kind: bdt, architecture: {max_iter: 1000, eval_every: 25, max_depth: 4}}
training: {negatives_per_positive: 20, batch_size: 65536, seed: 0, validation_events: 20000000}
```

Trees are added `eval_every` at a time; each step scores the validation events and saves
`best` on a lower log loss. One process only (no `torchrun`); the OpenMP threads follow
`OMP_NUM_THREADS`. `experiment.json` records the sample (`training_sample`: events, positives,
negative fraction and weight). The checkpoint stores the record layout (`stream_position` in
`model.json`); `predict` maps design-point θ and φ back to physical record columns with the
data's affine normalization.

## Why no stream CNP

A CNP predicts a target event from a context set of labeled events of the same task; here a
task is one design point, and the context tells the decoder which one it faces. Ungrouped
events of a homogeneous stream form a single task: a random context is independent of the
target given the inputs, so the optimal decoder ignores it and the CNP reduces to the MLP.
Grouping the stream (by position cells or a task id) would bin the very data the stream is
meant to use unbinned.
