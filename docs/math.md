# Math

The formulas behind the losses, the rate estimates and the GP stages. Derivations: the
[RESuM paper](https://openreview.net/pdf?id=lqTILjL6lP).

## Problem

Design point k, event i: `X_ki ~ Bernoulli(t(θ_k, φ_ki))`. Count `m = Σ_i X_ki`, raw rate
`y_raw = m/N`. Target: the marginalized rate `t̄(θ) = ∫ t(θ,φ) g(φ) dφ`, with `g(φ)` the
event-parameter distribution, and `θ* = argmin t̄`. For rare events (`m ≪ N`,
`m ≈ Poisson(N t̄)`) `y_raw` is coarse and noisy at small N. That noise is why there is a
surrogate.

## Event model

Predicts `p_ki ≈ t(θ_k, φ_ki)` from a single logit (cnp, mlp, field, transformer) and is
trained with BCE on X, i.e. the Bernoulli NLL. p depends only on (θ, φ), so the same model
serves every fidelity. Rate per design point: `ȳ(θ_k) = (1/N) Σ_i p_ki`.

## Legacy two-output CNP

`legacy_cnp` (paper code) outputs `(μ, s_raw)`; both objectives first apply the
logistic-normal moment approximation used by RESuM:

```text
s     = 0.1 + 0.9 softplus(s_raw)
d     = sqrt(1 + 3 s² / π²)
p̄     = sigmoid(μ / d)
v     = p̄(1-p̄)(1-1/d)
s̄     = softplus(v) + ε
```

`theory-truth` (default) follows the probability model:

```text
L_theory = -mean[X log p̄ + (1-X) log(1-p̄)]
```

For binary `X`, this is `-log ∫ Bernoulli(X | p) q(p) dp`, because the Bernoulli likelihood
is linear in `p` and therefore depends on `q` through `E[p]=p̄`. Use it when `β` is meant to
estimate the physical event probability.

`practice-truth` reproduces what the legacy Python code actually executed:

```text
L_practice = -mean[log Normal(X | p̄, s̄)]
```

This treats `{0,1}` observations as points under a continuous Normal density. It is kept for
reproduction only: it is not a Bernoulli likelihood and can behave differently for rare
events. Loss values of the two objectives are not comparable. The predicted mean `β = p̄` is
shared by both and bounded to `[0,1]`. Under theory-truth the two outputs only enter through
the effective logit `μ / d`, which is why the single-logit `cnp` is preferred
([models.md](models.md#legacy-two-output-cnp)).

## Route A: Bayesian last layer

Laplace posterior over the final linear layer's weights, features fixed:
`H = Σ p(1-p) f̃ f̃ᵀ + λ I` over the training events. Weight draws -> p draws per event ->
rate draws per design point. Details: [last-layer.md](last-layer.md).

## Route B: multi-fidelity GP

Kennedy–O'Hagan recursion `f_{l+1}(θ) = ρ_l f_l(θ) + δ_l(θ)`, `δ_l ~ GP`, scalar `ρ_l`. For
two levels the joint covariance is

```text
Σ = [[K_LL,      ρ K_LL         ],
     [ρ K_LL,    ρ² K_LL + K_δ  ]]
```

Levels come from the config, lowest first. Paper recipe: LF ȳ -> HF ȳ -> HF y_raw, given as
separate datasets, not stacked. Posterior at `θ*`: `μ* = K(θ*,Θ) K(Θ,Θ)⁻¹ y`,
`σ²* = K(θ*,θ*) − K(θ*,Θ) K(Θ,Θ)⁻¹ K(Θ,θ*)`. Config and outputs: [gp.md](gp.md).

## Binomial count GP

Latent GP on the logit: `m_k ~ Binomial(N_k, sigmoid(f(θ_k)))`, Laplace posterior. Optional
covariate z (logit of an event model's mean p): `logit p = b z + c + δ(θ)` with b, c
Gaussian and marginalized. Its NLL is a count probability mass, not a density.

## Active learning

IVR (exploration): `θ_new = argmin_θ ∫_Θ σ²(θ' | data ∪ θ) dθ'`. Tractable form for an RBF
kernel with MC points `θ_i`: `I(θ) ≈ (1/N) Σ_i k²(θ_i, θ) / σ²(θ)`. EI for exploitation.
Constraints enter as penalties that set the acquisition to zero outside the feasible region.

## Mixup

Class-aware mixup (`core/mixup.py`): `λ ~ Beta(α, α)`, paper α = 0.1,
`x̂ = λ x_i + (1−λ) x_j`, `ŷ = λ y_i + (1−λ) y_j`. Needed at paper-like imbalance
(~1 : 5·10⁴), not at synthetic rates (10–30 % positives). Settings: [training.md](training.md).

## Calibration target

Held-out ±1/2/3σ coverage ≈ 68.27 / 95.45 / 99.73 %. The paper reached 69 / 95 / 100 % with
the CNP levels and 12 / 24 / 47 % without them, the evidence for the multi-level design.
