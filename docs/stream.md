# Training on an event stream

`python -m core.surrogates stream CONFIG.yaml`

The MLP or the neural field can train on a large set of ungrouped events, e.g.
a homogeneous simulation campaign, instead of on events grouped by design point
(`core/surrogates/stream_training.py`, config `StreamRunConfig`). Only the MLP and the
field have stream training so far. The CNP, transformer and BDT could be added: the CNP
needs a rule that forms its context sets from the stream (today a context is the events of
one design point), the transformer only the record-to-input mapping, and the BDT a
subsample in memory, since boosted trees do not learn batch by batch.

```yaml
model: {kind: mlp, architecture: {hidden_dims: [256, 256, 256]}, last_layer: {}}
dim_theta: 2                     # input widths of the design-point data it is evaluated on
dim_phi: 3
mlp_position: {phi_columns: [0, 1]}   # phi columns added to theta = record position (optional)
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

- `kind: mlp`: an MLP on all W normalized record columns, any layout. `mlp_position` names
  the φ columns added to θ (none: θ alone is the position, all φ columns follow).
- `kind: field`: the neural field (point clouds); its `position` and `direction` settings
  define the layout, e.g. `dim_theta: 3`, `dim_phi: 6`, position φ[0:3], direction φ[3:6]
  gives records x, y, z, px, py, pz, label.

The event source implements the `EventSource` protocol (`n_events`, `n_positive`,
`normalization` of the W input columns, `n_batches`, `batches`, `load`); its reading and
shuffling options are its own and go into `event_source_options`. The model fixes its own
units from the store's train normalization (`set_stream_normalization`). Checkpoints are
selected by the Bernoulli log loss on the validation events. With `last_layer`, the selected
checkpoints get the Laplace posterior from one streaming pass over all training events.
Several GPUs: `torchrun` (DistributedDataParallel). bf16 autocast and `torch.compile` are
options. Run outputs label the stream metrics with fidelity `stream`.

The checkpoints are ordinary field checkpoints (the stream MLP is a field without
encodings). `predict` evaluates them on design-point data: θ, offsets and other φ columns
are mapped into the model's stream units, so no equal normalization is required.
