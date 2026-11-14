# Event models

The event model kinds behind one API (`core.surrogates`).

`core.surrogates` is the model-independent API for the single-logit CNP, MLP, neural
field, feature-token transformer and histogram boosted decision trees. All consume
`StandardBatch` and produce `EventPrediction`: event logits, probabilities, and
design point means. Uncertainty is optional: the MLP, CNP, field and transformer can add a
Bayesian last layer ([last-layer.md](last-layer.md)); the BDT and legacy CNP have none.

## Python API

```python
from core.surrogates import build_surrogate, Episode, evaluate_surrogate, load_surrogate
from schemas.surrogates import NeuralTraining

# Only the model spec changes between the neural architectures.
model = build_surrogate({"kind": "cnp"}, dim_theta=3, dim_phi=6, seed=0)
training = NeuralTraining(
    sampling={"strategy": "positive_quota", "positive_fraction": 0.05},
    weighting={"strategy": "sampling_correction"},
)
report = model.fit(
    train_batch, training,
    validation=Episode(validation_context, validation_targets),
    selection="design_point_rate_mae",
    checkpoints="outputs/my_run/checkpoints",
)
prediction = model.predict(validation_targets, context=validation_context)
metrics, arrays = evaluate_surrogate(model, validation_targets, context=validation_context)
restored = load_surrogate("outputs/my_run/checkpoints/best")
```

Construct `kind="mlp"`, `kind="transformer"` or `kind="bdt"` using the same factory.
BDT uses `TreeTraining()` and requires the optional `.[bdt]` dependencies.
The CNP requires context observations from the same design points. Other models ignore
context (they never see context labels), so the same episodes can be passed to them.
The runner gives them no context: they train on every drawn event and are scored on all
events of each design point, CNPs on the events after the context pool
([evaluation.md](evaluation.md)). Their training configs reject `context`, `targets`
and `mixup.mix_context` ([training.md](training.md)). All prediction shapes are
`[design_point, event]`; the mean is an unweighted average over exactly those events.

Architecture options live inside the model spec: `encoder` for CNP and
`architecture` for MLP/transformer/BDT. MLP architecture exposes only hidden
widths and dropout, without unused CNP latent settings. Transformer defaults are the revised
122,113-parameter FT-style configuration. Neural initialization seed is explicit
in the factory; training seed controls episode sampling and dropout. BDT's seed
is in its architecture config.

`fit` returns history, the selected step/tree count, and a sampling audit. It
restores the selected model. The optional checkpoint directory contains both
`best/` and `final/`. Without validation, the last step is selected. Each neural
fit creates a fresh optimizer and continues from the module's current weights;
each BDT fit resets the estimator. This is not optimizer-state resume support.

## Neural field (kind: field)

The neural field is meant for point-cloud data: events with continuous coordinates (any
dimension) and optionally a 3-D direction vector, e.g. particles or rays with a start point
and a direction. The spherical-harmonic direction encoding needs exactly 3 direction
columns. For other event inputs use the MLP, CNP, transformer or BDT, which take any number
of θ and φ columns.

```yaml
model:
  kind: field
  architecture: {hidden_dims: [256, 256, 256]}
  direction: {phi_columns: [3, 4, 5], degree: 4}
  encoding: {kind: fourier, columns: [0, 1, 2], n_frequencies: 8, max_frequency: 8.0}
  last_layer: {}                               # optional
```

A context-free model like the MLP, but on encoded coordinates (`core/bernoulli_field.py`).
`encoding.columns` names what is Fourier-encoded: `theta` (the default) or a list of phi
columns, taken as they are; theta is then not an input. φ holds each event's own inputs
(e.g. the absolute emission point), and the toolkit never adds θ to them: an application
that stores events relative to their design point makes them absolute in its dataset
provider. With the emission point in φ the field sees the emission point and never the grid
of design points. The encoded columns (in normalized units) get the raw value plus sin/cos
at `n_frequencies` octave-spaced frequencies up to `max_frequency` cycles per unit; the
finest half-period it can represent is `1 / (2 max_frequency)`. `encoding: {kind: none}`
leaves them unencoded. `direction` columns are mapped back to physical units with the
provider's affine normalization (`x_n = (x - offset) / scale`, stored in the checkpoint),
normalized to unit length and encoded by real orthonormal spherical harmonics up to
`degree` ((degree + 1)² features). All other phi columns enter unencoded. The former
`position` setting (θ + φ offsets) was removed on 2026-11-13; configs and checkpoints with it
fail with a hint. Training, selection, inference, the GP levels and
the Bayesian last layer work as for the MLP. `architecture` is an MLP (`hidden_dims`) or,
given transformer keys (`token_dim`, `n_heads`, ...), the tabular transformer with one token
per encoded column; the stream transformer (`docs/stream.md`) is this field without encodings.

## Legacy two-output CNP

Prefer the single-logit `cnp`. Under
`theory-truth` the two outputs only enter through one effective logit
`mu_logit / d(sigma)`, so the second channel adds no representable predictions
and is not identified by binary labels. `legacy_cnp` is kept to reproduce the
original RESuM objective and older checkpoints.

`model.kind: legacy_cnp` wraps the existing `build_cnp` architecture without
changing its decoder or likelihood definitions. Set `training.loss` explicitly:

- `theory-truth`: Bernoulli NLL on the effective logit computed from both outputs.
- `practice-truth`: the original Gaussian NLL on the transformed mean and scale.

Single-logit models keep `training.loss: bernoulli` (the default). Mismatched
model/loss settings fail validation. Legacy losses require `focal_gamma: 0`.
Natural sampling (optionally with explicit class weights), class-aware mixup
alone, `objective.strategy: real_plus_mixup`, and real-target oversampling are
supported. For oversampling, set

```yaml
sampling: {strategy: positive_quota, positive_fraction: 0.05}
weighting: {strategy: sampling_correction}
```

Each event's legacy loss (theory-truth BCE or practice-truth Gaussian NLL) is
multiplied by its per-design-point importance weight p/q or (1-p)/(1-q) before the
event mean, so the expected loss matches uniform real targets. For theory-truth
this is an unbiased estimate of the natural Bernoulli risk. Practice-truth is not
a Bernoulli likelihood, so the weights only restore its natural-sampling objective. For the combined objective,
**both** branches use the selected legacy loss and one optimizer update:
`mean(real legacy loss) + mixup_loss_weight * mean(mixed legacy loss)`.

The `practice-truth` combined history uses `training_real_nll` and
`training_mixup_nll`; `theory-truth` retains the BCE component names. These replace
the BCE-only restriction of the real-plus-mixup objective ([training.md](training.md)) specifically for the legacy adapter.
Both output channels receive gradients. The Beta mixing parameter and relative
loss weight remain separate settings.

Shared predictions use `resum_binary_logits` (whose sigmoid is the transformed
mean), not the historical `predict_beta` helper's sigmoid of the raw mean logit.
The transformed scale is preserved as `EventPrediction.legacy_scale` and in
saved evaluation arrays under `legacy_scale`. It is the original scale proxy,
not a calibrated uncertainty estimate or an MFGP noise variance. Shared metrics,
PR curves, mean plots and MFGP preparation use the effective probabilities.
Validation labels stay real and binary even with the Gaussian training loss.
The common `bernoulli_log_loss` metric remains Bernoulli log loss for comparison;
it is not the Gaussian training objective.

Select the combined objective explicitly using the settings above. New checkpoints
roundtrip both output channels through the standard loader. Importing historical
two-output checkpoint files still uses `core.training.load_checkpoint`; the
single-logit legacy importer does not silently convert those files.
