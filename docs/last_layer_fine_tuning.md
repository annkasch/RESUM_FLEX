# Mixup followed by last-layer fitting

The interchangeable neural-model API supports a second training stage through
`training.fine_tuning`. The first stage retains the configured sampling and loss.
The second stage reuses the original training events, draws real contexts and
real targets naturally, and optimizes unweighted Bernoulli cross-entropy. It
freezes all parameters except the architecture's final linear projection. Frozen
features run in evaluation mode, so dropout does not change their representation.
A fresh optimizer is created for this stage.

For the two-output CNP, the final projection produces both original decoder
channels. Both are retrained through the existing marginalized Bernoulli score
calculation (`theory-truth`), including when the first stage used `practice-truth`.
The architecture and event-score averaging interface stay the same. This does
not identify or validate the second channel as an uncertainty estimate.

```yaml
training:
  # Existing first-stage settings go here.
  fine_tuning:
    n_steps: 2000
    learning_rate: 0.0003
    eval_every: 250
    n_events: 512
```

Omit `fine_tuning` for single-stage training. CNP, two-output CNP, MLP and tabular
transformer components all expose the final projection. `n_events` defaults to
inheriting the first stage's episode size; it must fit the available data. Context
sizes and voxel batch size inherit the first-stage settings. Steps count fresh
randomly sampled episodes, rather than full dataset passes.

The trainer selects checkpoints separately in each stage using the configured
validation metric. It starts the second stage from the selected first-stage
checkpoint. The second-stage selection does not compete with the first stage:
its result can therefore be worse, which is why both predictions are reported.
Without validation, each stage uses its final training step.

Saved checkpoints:

- `checkpoints/pretraining`: selected model before last-layer fitting.
- `checkpoints/best`: selected fine-tuned model, consumed by the existing MFGP stage.
- `checkpoints/final`: model at the last fine-tuning step.
- `checkpoints/stages/pretraining/{best,final}`: first-stage artifacts.

History records stage-local and cumulative steps. Sampling audits record the
actual real-event prevalence and the names of trainable parameters. The usual
model save/load and `prepare_surrogate_datasets` interfaces remain usable.

## LF1500 development and LF1000 external test

Run `notebooks/lar_optical_map_fine_tuning.ipynb`. It prepares the data, shows a
count table, trains both stages, and displays before/after means, residuals and
precision–recall curves. The notebook creates a new run directory each time.

- `config.optical.fine_tuning.data.yaml` controls folders, feature construction,
  normalization and the spatial split. Both budgets include zero-hit files.
- `config.optical.fine_tuning.yaml` controls the architecture, training schedule,
  checkpoint selection and final test evaluation.

The default development split is 80% LF1500 training and 20% LF1500 validation
voxels, grouped by filename centers. LF1000 supplies the external test. Test
locations within 5 mm of any development location are excluded, and the exclusions
are saved in `test_source_manifest.json`. Tests use normalization fitted only on
LF1500 training. The expected simulation budgets and single-primary event contract
are checked against each file's contents. Source-file identity is checked; simulator
random-seed independence cannot be inferred from the available file metadata.

The prepared LF1000 batch is loaded by `run_experiment` only after fitting and
selection. `test_fidelities` defaults to an empty list in ordinary RESuM runs;
requesting `[lf]` explicitly enables this final evaluation. LF1000 is never assigned
an HF modeling role.
All before/after results use the same fixed real context and target events. Rates
and PR metrics use the target population after 64 context events are removed.

The notebook also fits the existing three-level MFGP in log space: selected
fine-tuned LF means, selected fine-tuned HF means, and observed HF target fractions.
N5000 zero-hit and nonzero folders supply 10 HF training voxels and the remaining
HF validation voxels (`split.hf_train_count`). The CNP trains and selects only on
LF1500. `normalization.fit_fidelities: [lf]` preserves LF-only normalization while
applying the same transforms to HF. External test spatial exclusion includes HF
locations as well as LF development locations.

The notebook displays GP input scores, means, coverage, metrics and configured
axis/plane projections. Removing `mfgp` restores the CNP-only runner. No change to
the GP likelihood is implied; lognormal observation bands and projection bands
(latent GP plus binomial counts) retain their existing different interpretations.
Setting `RUN_TRAINING = False` displays the latest run with saved MFGP metrics;
older CNP-only runs are not treated as completed GP runs.

## Assess the experiment

Compare mean bias (predicted minus observed), voxel MAE, spatial trend and average
precision before/after on the external test. Inspect the LF1500 validation history
for convergence. The 12,000-step mixup stage and 2,000-step output-layer stage are
starting settings, not an established optimum. Fine-tuning cannot recover missing
features or add independent positives, and does not guarantee probability calibration.
Do not tune the schedule on the LF1000 test results.
