# Spatial projections

Posterior marginalizations and predictive projections of a fitted GP over physical coordinates.

## Shared engine and adapters

`core/spatial_projections.py` implements the integration grid, physical-volume
averages, individual-design-point mixtures, and all x/y/z and xy/xz/yz projections.
`viz/spatial_projections.py` renders the common artifact format. The old MFGP
entry points and `MFGPProjectionConfig` remain compatible aliases/wrappers of
`schemas.projections.ProjectionConfig`. YAML `projections.enabled` controls this step. Saved outputs are `projections/` (latent mean) and
`projections_observed_fraction/` (individual observations), each with curves,
planes, NPZ arrays and metadata. Predictive projections have validation inclusion
counters. They use the saved HF budget when unique; mixed budgets require an
explicit `target_events`. No implicit median budget is substituted.

To add a new GP, implement a projection adapter's `prepare(physical_coordinates)`
method returning an object with `mean`, `metadata`, and `sample(n_draws, rng)`.
The returned array must have shape `(draws, locations)` and preserve joint spatial
correlations. Values are in physical response units, with no observation noise.
The adapter owns coordinate scaling, fidelity selection, and the latent link.
For Gaussian latent posteriors, `GaussianResponse` supports identity, log and
logit links. Other posteriors can supply their own joint-draw implementation.
Register an adapter factory with `register_projection_adapter(name, factory)`;
no integration or plotting code changes are necessary.

```python
from core.projection_adapters import projection_adapter, BinomialObservation
from core.spatial_projections import project_response
from schemas.projections import ProjectionConfig

adapter = projection_adapter("mfgp", fitted_mfgp)
settings = ProjectionConfig(quantity="observed_fraction", target_events=5000)
arrays, metadata = project_response(
    adapter, physical_training_centers, settings,
    observation_model=BinomialObservation(5000),
)
```

For `latent_mean`, omit the observation model: each joint response draw is
averaged before its quantiles are computed. For `observed_fraction`, the engine
retains omitted-coordinate variation and applies a separate observation model.
The MFGP adapter does not include learned Gaussian observation noise. Invalid
binomial probabilities fail rather than being clipped. Uniform midpoint-cell
weights and dense joint covariance are currently used; grid size is capped to
control memory. To compare models, use the same explicit physical box, grid,
weights, and counting budget. Training convex hulls can differ between datasets.

Projection figure selection is controlled by `mfgp.projections.plots`. The default, when omitted, is:

```yaml
plots: [projected_axes, projected_planes]
```

These are the axis and plane predictions with observed data and coverage counters.
Add `marginalized_axes` and/or `marginalized_planes` for spatially averaged
probabilities. Any subset and order is supported; `plots: []` selects none.
`enabled: false` disables the whole step. The GP runner uses
`ensure_selected_projections`; it computes only requested quantities and
renders only requested figures. Existing unselected files are retained but not
displayed. The `quantity` field is retained for low-level, single-quantity calls;
the high-level runner derives quantity from `plots`.

## Spatial posterior marginalizations

`mfgp.projections.enabled: true` adds highest-fidelity spatial projections after
GP fitting; `ensure_selected_projections` also generates them from an existing saved
model, without retraining or loading raw data. The implementation follows
[RESOLVE's posterior-draw averaging recipe](https://github.com/annkasch/resolve/blob/4219c8c101d30acc38401465c5dc17ba88dcce1a/resolve/regression_models/bayes_pce_multi_fidelity_visualizer.py#L164),
not its single-prediction bin-percentile method, whose spread describes spatial
variation rather than uncertainty in a population average.

All three coordinates are queried jointly at the highest GP fidelity. Draws use
the full latent posterior covariance, excluding likelihood noise, conditional on
fitted hyperparameters. In log mode, each draw is exponentiated before averaging.
Uniform midpoint cells approximate physical-volume averages. Curves integrate
out two coordinates; plane maps integrate out one. Each remaining bin normalizes
by its valid cell count, so this is a conditional average, not an unnormalized
integral. Posterior quantiles of the averaged draws define 68.27%, 95.45%, and
99.73% pointwise equal-tail intervals (not simultaneous bands). The plotted mean uses analytic pointwise latent
moments averaged with the same weights. It is not the median. Observation-predictive
coverage plots remain separate and retain their previous noise convention.

Example projection settings (`mfgp.projections`):

```yaml
mfgp:
  projections:
    enabled: true
    domain: training_convex_hull
    grid_steps: 12
    n_draws: 8192
    seed: 42
    axis_labels: [x, y, z]
    units: m
```

The default domain is the convex hull of LF/HF **training** centers, mapped back
using the saved normalization. It is empirical support, not a detector geometry
model; it cannot represent concavities or holes. `domain: box` requires explicit
`bounds: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]` in physical units. A box must
lie within the intended volume. Optional bounds also restrict a hull domain.
No bounds are inferred from validation labels or validation positions. Grid
midpoints outside the domain are excluded; unsupported projected cells remain
blank, never filled with zero. Defaults are exploratory: check grid/draw
convergence before interpreting fine structure or extreme tails. Dense joint
covariance scales quadratically in valid grid points; grid_steps is capped at 16.

Artifacts under `RUN_DIR/mfgp/projections/`:

- `curves.png`: three coordinate curves with all three bands.
- `curves_observed.png`: the same curves with individual HF training and
  validation target fractions overlaid and identified separately.
- `planes.png`: xy/xz/yz means and three interval-width maps.
- `planes_observed.png`: mean maps with observed target fractions as colored
  markers, on the same color scale; training circles and validation diamonds.
- `projections.npz`: physical grids, domain mask, projected means/interval bounds,
  and unchanged raw target fractions/positions from the saved run.
- `metadata.json`: configuration, model/input hashes, numerical jitter, domain,
  weighting and uncertainty definitions. Matching artifacts are cached.

Overlaid observations are individual design point measurements at different omitted
coordinates, not volume-marginalized measurements. Their scatter includes spatial
variation and counting noise. Their inclusion in the marginal bands is **not** a
coverage test. Three-sigma tail quantiles are Monte Carlo estimates, not exact.

## Predictive projections for observed design point fractions (projection 3)

`mfgp.projections.quantity: observed_fraction` selects this mode; `latent_mean` remains available for uncertainty in spatial averages. Predictive
projections retain variation between locations: for each posterior draw and each
retained coordinate bin, uniformly select a valid grid location in the omitted
coordinates, then use a binomial count with its latent probability. The resulting
`m/N` samples determine discrete equal-tail 68.27/95.45/99.73% quantiles. No
spatial averaging occurs before these quantiles. The central curve remains the
population mean. Both 1D and plane projections support this mode.

`target_events: null` (default) recovers N from the saved evaluation set when all its
design points have the same event count; otherwise set it explicitly. A different N describes a different simulation
budget and should not be interpreted as matched to the overlaid measurements.

Only latent GP uncertainty is sampled; fitted Gaussian observation noise is not
added alongside binomial noise. Treating the latent log-GP as a probability model
is approximate because fitting used smoothed rates. Any sampled probabilities
outside [0,1] cause an error rather than clipping. The log-GP does not enforce
probability support; substantial excursions need a bounded model.

These are **unconditional** predictions, allowing zero counts, over a uniform
spatial population in the configured domain. Current nonzero-selected HF files
and their spatial sampling do not represent that population exactly. Overlays
remain descriptive, not evidence of nominal predictive coverage. No rejection
sampling or conditioning on positive counts is silently applied.

Artifacts are written under `mfgp/projections_observed_fraction/`, keeping the
latent-mean artifacts separate. Grid/draw resolution and tail Monte Carlo error
remain relevant; discrete intervals may contain more than nominal probability.

Predictive observed-overlay figures include per-projection validation counters
for all three bands, alongside their nominal probabilities. Counts use interpolated displayed bounds for curves and the saved grid cell
for planes (last edge inclusive), with inclusive interval bounds. Points outside
projected edges, in undefined cells, or with nonfinite values are excluded and
reported; denominators count only evaluable validation points, never training
points. `coverage_counts.json` stores the counts for all six projections. These
are descriptive interval-inclusion rates, not proof of population calibration
for the spatially/nonzero-selected HF validation sample. No extrapolation is used.

The 1D predictive displays now connect grid-center estimates linearly; the first
and last values extend only to their own cell edges. Validation counters use
exactly those interpolated bounds, with no extrapolation beyond the projection
range. Plane counters retain grid-cell lookup. Observations use circle markers.
Points outside training-derived projection bounds remain visible but are excluded
from inclusion denominators. Rendering does not change the sampled predictions.
