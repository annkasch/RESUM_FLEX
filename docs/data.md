# Data

What goes into the toolkit: `StandardBatch`, input scaling, dataset providers and unequal event counts.

## `StandardBatch`

The primary pipeline carrier. Three modalities are supported via the `mode` flag:

| `mode` (`InputMode`) | `theta` | `phi` | `labels` (X) |
|---|---|---|---|
| `InputMode.FULL` | `[B, D_θ]` | `[B, N, D_φ]` | `[B, N]` binary {0,1} |
| `InputMode.EVENT_ONLY` | `None` | `[B, N, D_φ]` | `[B, N]` binary |
| `InputMode.DESIGN_ONLY` | `[B, D_θ]` | `None` | `[B, N]` binary |

`B` = number of trials; `N` = events per trial; `D_θ`, `D_φ` ≥ 1 (arbitrary).
Optional field `beta: [B, N] ∈ [0, 1]` is filled by the CNP.

```python
from schemas.data_models import StandardBatch, InputMode
batch = StandardBatch(
    mode=InputMode.FULL,
    theta=theta_arr,        # (B, D_θ)
    phi=phi_arr,            # (B, N, D_φ)
    labels=labels_arr,      # (B, N), values in {0, 1}
)
```

The validator cross-checks `mode` against the presence of `theta` / `phi`
and verifies all batch dims agree, so malformed batches fail at construction.

## Build a batch from your arrays

The user-facing flow is the **bring-your-own-data** path. The synthetic
generator (`for_scenario("S1"..."S8")`) is for validation / unit tests.

The package consumes `StandardBatch` objects. Load your simulation data
(HDF5, npz, CSV, ROOT, …) into numpy arrays, then instantiate:

```python
import numpy as np
from schemas.data_models import StandardBatch, InputMode

# Example: load from your file format
arrays = np.load("my_lf_simulation.npz")
theta_lf  = arrays["theta"]      # shape (n_trials, D_θ)
phi_lf    = arrays["phi"]        # shape (n_trials, n_events_per_trial, D_φ)
labels_lf = arrays["X"]          # shape (n_trials, n_events_per_trial), {0, 1}

lf_batch = StandardBatch(
    mode=InputMode.FULL,
    theta=theta_lf,
    phi=phi_lf,
    labels=labels_lf.astype(np.int8),
)
```

For event-only or design-only modalities, set the absent component to
`None` and use the matching `InputMode`:

```python
event_only = StandardBatch(mode=InputMode.EVENT_ONLY, theta=None, phi=phi, labels=X)
design_only = StandardBatch(mode=InputMode.DESIGN_ONLY, theta=theta, phi=None, labels=X)
```

The validator throws on shape / mode mismatches at construction time.

## Normalize before you build the batch

The CNP encoder is an MLP that fits the **raw numerical input**. When
`θ` (or `φ`) components have ranges that differ by ≥ ~10× — *whatever
the units* — gradient imbalance makes the encoder *scale-blind*: it
locks onto the high-magnitude dimension and ignores the small one
(often by a factor of 100× in effective gradient flow, even after
long training). The MFGP can compensate via ARD lengthscales; the
CNP cannot.

**Always normalize before building the batch when feature ranges
differ.** Use `core.MinMaxScaler` and persist it alongside your CNP /
MFGP checkpoint so predictions can be inverse-transformed back to the
original units later. **Any numerical ranges are supported** — the
scaler simply maps each feature's `[low, high]` linearly onto
`[-1, 1]` (or any target interval you choose).

Three equivalent ways to construct it:

```python
from core import MinMaxScaler

# 1. Preferred — known per-feature bounds (any numerical ranges).
theta_scaler = MinMaxScaler.from_bounds(
    low=theta_low,             # 1-D array of length D_θ
    high=theta_high,            # 1-D array of length D_θ
)

# 2. Fit from data when bounds aren't known a priori.
theta_scaler = MinMaxScaler.fit(theta_train)

# 3. Pick a different output interval (default is [-1, 1]).
theta_scaler = MinMaxScaler.from_bounds(
    low=theta_low, high=theta_high,
    target_low=0.0, target_high=1.0,
)
```

Concrete examples — the same call works for any ranges:

```python
# Physical units that span vastly different magnitudes:
sc = MinMaxScaler.from_bounds(low=[500.0, 0.0],   high=[3000.0, 1.0])

# Negative ranges, mixed scales:
sc = MinMaxScaler.from_bounds(low=[-50, 1e-6],    high=[50, 1e3])

# 4-D design space:
sc = MinMaxScaler.from_bounds(low=[0, 0, 0, -π],  high=[10, 100, 1, π])
```

Apply, build the batch, train, predict, invert:

```python
theta_scaled = theta_scaler.transform(theta_raw)            # forward
lf_batch = StandardBatch(mode=InputMode.FULL,
                          theta=theta_scaled,
                          phi=phi_lf,                        # use a separate scaler
                          labels=labels_lf.astype(np.int8))   #   for φ if needed

# At predict time, scale the query with the *same* scaler:
mu, var = mfgp.predict(theta_scaler.transform(theta_query_raw))

# Inverse-transform any θ-shaped output (e.g. an AL record) back to original units:
theta_next_raw = theta_scaler.inverse_transform(record.theta_next.reshape(1, -1))[0]
```

If you skip normalization on imbalanced inputs, the framework emits a
`ScaleImbalanceWarning` at `StandardBatch` construction. The threshold
is a 10× per-feature range gap — see `schemas.data_models.SCALE_IMBALANCE_THRESHOLD`.

## Datasets and providers

Training, inference and the GP stage read data only through a dataset provider
(`core/surrogates/datasets.py`). A run config names it by import path; without one,
`NpzBatches` reads `batches/<partition>/<fidelity>.npz` (synthetic data, tests):

```yaml
data_directory: outputs/my_prepared_data
data_provider: my_package.provider:MyDataset    # constructed as MyDataset(data_directory)
data_config: my_data.yaml                        # optional: provider.prepare() rebuilds if outdated
partitions: {train: train, validation: validation, test: test}   # provider partition names
```

A provider implements `partitions()`, `load(partition, fidelity, labels=True)` returning
`(StandardBatch, provenance)` (with `labels=False` also for inputs without outcomes; the
labels are then a zero placeholder that is never scored), `theta(partition, fidelity)`
returning normalized theta and design point ids, `coordinates(partition, fidelity)` and
`normalization()`. Optionally a classmethod `prepare(data_config, data_directory)` rebuilds
the prepared data and returns a record that the run stores in `experiment.json`. GP levels
and evaluation sets name provider partitions directly. Application packages ship their own
provider and configs.

Two settings make large validation sets cheap. `selection_max_design_points: N` selects
checkpoints during training on a seeded subset of at most N validation design points; the full
validation and test sets are still evaluated once after training. `evaluation_threads: N`
runs evaluation passes with N torch CPU threads; results are identical.

## Unequal event counts per design point

Design points may contain different numbers of events. Data preparation pads every design point
of a split/fidelity group to the largest count and stores the true counts as
`event_counts` (`[B]`) in `StandardBatch.event_counts` and in
`metadata.json`. Padded positions have zero labels and features.
Groups with equal counts keep the previous rectangular format, so existing
prepared datasets and checkpoints are unaffected.

- **Training** weighs design points equally: design points are drawn uniformly and each episode
  samples context and targets only from that design point's own events. `n_events` must
  fit the smallest training design point. This holds for natural sampling, positive quota
  (pool prevalence uses the design point's own events) and class-aware mixup. BDT fits all
  valid events and drops padding.
- **Context/target splits** (`split_context_target`, evaluation manifests) use a
  fixed context size; each design point's target is its remaining events, so target
  counts differ per design point and are carried as `event_counts` / `target_counts`.
- **Design point means** (`EventPrediction.mean`, `StandardBatch.design_point_mean`, observed
  `m/N`) divide by each design point's own count. Event-level metrics (log loss, Brier,
  AP) use valid events only.
- **MFGP**: `training_arrays.npz` stores `events_k` per design point of level k, and log
  smoothing `(m + a)/(N + 2a)` uses each design point's own N. The GP still has one
  constant noise per level, so design points with smaller N are noisier than the model
  assumes. Observed-fraction projections
  need an explicit `projections.target_events` when validation counts differ.
- The legacy `core.training.train_cnp` loop, `cnp_trial_predictive`, `evaluate_bernoulli`, `score_bdt` and the legacy MFGP
  dataset helpers reject unequal counts with an error; use `core.surrogates`.
