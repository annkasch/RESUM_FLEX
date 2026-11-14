# Training on an event stream

`python -m core.surrogates stream CONFIG.yaml`

An event model can train on a large set of ungrouped events, e.g. a homogeneous simulation
campaign, instead of on events grouped by design point (`core/surrogates/stream_training.py`,
config `StreamRunConfig`). Kinds: `mlp`, `field`, `transformer` (gradient steps on batches)
and `bdt` (an in-memory case-control sample). There is no stream CNP (below).

```yaml
model: {kind: mlp, architecture: {hidden_dims: [256, 256, 256]}, last_layer: {}}
dim_theta: 2                     # theta width of the evaluation views (optional; null/0 = none)
dim_phi: 3                       # = the stream record's input columns
training: {batch_size: 65536, n_passes: 10, device: cuda, precision: bf16}
event_store: /path/to/store
event_source: my_package.loader:MyEventStream   # required: the application's reader
event_source_options: {in_memory: true}         # passed to it unchanged
output_directory: runs/stream_mlp/seed0
```

The stream format (`core/event_stream.py`): float32 records [M, W + 1] = the W = `dim_phi`
event inputs in φ order, then the label bits. A stream model never uses θ: at evaluation on
design-point data it sees the φ columns of each event as they are, so the design-point data
must hold the same columns as the stream (e.g. the absolute emission point, not an offset
from the design point). Views with θ and event-only views (no θ input; EVENT_ONLY) are
scored alike, whatever `dim_theta` says: it only documents the evaluation views and may be
left out (`null` or `0`). On an event-only view the feature calibration covers the φ
columns only. Offsets, coordinate transforms and the like belong in the
application: its event source and its dataset provider deliver the same columns. Width
checks: the source's `normalization()` must cover W columns, and every record batch must be
[M, W + 1]; nothing checks what a column means.

- `kind: mlp`: an MLP on all W normalized record columns.
- `kind: transformer`: the tabular transformer on the same normalized columns, one token per
  column (`architecture` as for design-point training).
- `kind: field`: the neural field (point clouds); `encoding.columns` (φ columns, not `theta`)
  are Fourier-encoded, `direction` takes 3 φ columns, e.g. `dim_phi: 6`, `encoding.columns:
  [0, 1, 2]`, direction φ[3:6] gives records x, y, z, px, py, pz, label.
- `kind: bdt`: boosted trees on the W record columns in physical units. `training` takes
  other settings, see below.

`record_position` / `mlp_position` (φ columns added to θ) were removed on 2026-11-13;
configs and checkpoints with them fail with a hint.

The event source implements the `EventSource` protocol (`n_events`, `n_positive`,
`normalization` of the W input columns, `n_batches`, `batches`, `load`); its reading and
shuffling options are its own and go into `event_source_options`. The neural models fix their
own units from the store's train normalization (`set_stream_normalization`). Checkpoints are
selected by the Bernoulli log loss on the validation events. With `last_layer` (not bdt),
the selected checkpoints get the Laplace posterior from one streaming pass over all training
events.
The stream validation events also get the calibration per record column (`phi_j`,
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
data: each φ column is mapped from the data's normalization into the model's stream units,
so no equal normalization is required; θ is not used.

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
negative fraction and weight). The checkpoint marks the trees as stream-trained
(`stream_records: phi` in `model.json`); `predict` maps design-point φ back to physical
record columns with the data's affine normalization.

## Why no stream CNP

A CNP predicts a target event from a context set of labeled events of the same task; here a
task is one design point, and the context tells the decoder which one it faces. Ungrouped
events of a homogeneous stream form a single task: a random context is independent of the
target given the inputs, so the optimal decoder ignores it and the CNP reduces to the MLP.
Grouping the stream (by position cells or a task id) would bin the very data the stream is
meant to use unbinned.
