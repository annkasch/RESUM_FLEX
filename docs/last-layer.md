# Bayesian last layer (mlp, cnp, field)

```yaml
model:
  kind: mlp            # or cnp, field
  architecture: {hidden_dims: [256, 256, 256]}
  last_layer:
    prior_precision: evidence   # or a fixed number
    draws: 1000
    seed: 0
```

After training and selection, every saved checkpoint gets a Laplace posterior over its final
linear layer (`core.last_layer`, `core.surrogates.last_layer`); nothing is retrained. The
trained weights are the mean; the precision is the exact Bernoulli Hessian of that layer,
`H = Σ p(1-p) f̃ f̃ᵀ` over all natural training events (f̃ = penultimate features plus a bias
input), plus `λ I`. `prior_precision: evidence` picks λ by maximizing the Laplace marginal
likelihood of the training data; validation is not used. A CNP conditions on the same
seeded context split as evaluation (`validation_context_events`); only its target events
enter H. Context-free models use all training events. The posterior is stored as `checkpoints/<name>/last_layer.npz`, its summary in the
manifest's `metadata.last_layer`.

Predictions: `voxel_probability_draws` samples the weights and averages p over each voxel's
events per draw, so the uncertainty stays correlated across the voxel's events and does not
average down. For held-out splits the runner adds `voxel_predictive` to each metrics row:
counts `k ~ Binomial(n, p)` mixed over the draws (`core.gp_metrics.count_predictive_scores`),
with count NLL, coverage 1/2/3σ, CRPS, MAE/RMSE and interval scores as for the binomial GP,
and writes `<fidelity>_<split>_<checkpoint>_predictive.npz` and a coverage plot.

Limits: the feature layers are treated as fixed, so their uncertainty is not included, and
the network was trained by maximum likelihood (the prior is post hoc). Check coverage on
held-out voxels before relying on the error bars.
