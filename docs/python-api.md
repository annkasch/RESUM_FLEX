# Python API (configs, legacy CNP)

Loading configs in code and the legacy two-output CNP workflow (`core.training`).

## Configuration

Run configurations are YAML files validated by `schemas/surrogates.py`
(training: `load_surrogate_config`; standalone GP: `load_mfgp_config`).

```python
from schemas.surrogates import load_surrogate_config
cfg = load_surrogate_config("TRAINING.yaml")
print(cfg.training.learning_rate)
```

Legacy APIs still accept their typed configurations, constructed in code:

### Build legacy configs in code

```python
from schemas.config import EncoderConfig, CNPConfig, TrainingConfig

enc_cfg = EncoderConfig(type="mlp", latent_dim=64, hidden_dims=[128, 128], dropout=0.0)
cnp_cfg = CNPConfig(n_context_min=16, n_context_max=64,
                    objective="theory-truth")
train_cfg = TrainingConfig(n_steps=1500, learning_rate=1.0e-3, batch_size=16,
                           n_events_per_trial=128, seed=0)
```

### Override a single field

Validate updates through the shared training schema:

```python
from schemas.surrogates import NeuralTraining
cfg = load_surrogate_config("TRAINING.yaml")
custom_train = NeuralTraining.model_validate({
    **cfg.training.model_dump(), "n_steps": 3000, "learning_rate": 5e-4,
})
```

Pass `custom_train` to the shared model's `fit` method.

## Train the legacy CNP

`train_cnp` consumes a duck-typed *batch generator* — anything with the
shape

```python
class HasGenerate:
    mode: InputMode
    dim_theta: int | None
    dim_phi:   int | None
    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch: ...
```

For synthetic data this is `PseudoDataGenerator`. For an in-memory real
dataset, use the supplied `FixedBatchSource` adapter:

```python
from data import FixedBatchSource

sampler = FixedBatchSource(full_lf_batch)
```

Then:

```python
import torch
from core import build_cnp, train_cnp, save_checkpoint
from schemas.config import EncoderConfig, CNPConfig, TrainingConfig

torch.manual_seed(0)
enc_cfg = EncoderConfig(type="mlp", latent_dim=32, hidden_dims=[64, 64], dropout=0.0)
cnp = build_cnp(enc_cfg, dim_theta=sampler.dim_theta, dim_phi=sampler.dim_phi)

history = train_cnp(
    cnp, sampler,
    cnp_config=CNPConfig(n_context_min=32, n_context_max=96,
                         objective="theory-truth"),
    training_config=TrainingConfig(
        n_steps=1500, learning_rate=1e-3,
        batch_size=16, n_events_per_trial=128,
        eval_every=0,                    # ← 0 for real data; see note below
        seed=0,
    ),
)

save_checkpoint("results/cnp.ckpt", cnp,
                encoder_config=enc_cfg,
                dim_theta=sampler.dim_theta, dim_phi=sampler.dim_phi,
                history=history, metadata={"data": "my_simulation_v1"})
```

**Note on `eval_every`:** the in-loop MAE evaluation compares predicted
`β` against analytical `p` from `generator.truth`. Synthetic data has
that; real data does not. Set `eval_every=0` to disable it on real data,
and run your own held-out evaluation (CNP-only coverage below, or the MFGP coverage check in
[gp-api.md](gp-api.md)) after training.

The two `objective` values (`theory-truth`, `practice-truth`) are defined in
[math.md](math.md#legacy-two-output-cnp).

## CNP-only coverage check (no MFGP)

For pipelines without a fidelity tier (or as a sanity check before MFGP):

```python
from core import cnp_trial_predictive, split_context_target
from viz import plot_coverage_test

# Held-out HF batch
holdout = StandardBatch(mode=InputMode.FULL, theta=hf_theta_test,
                        phi=hf_phi_test, labels=hf_X_test.astype(np.int8))
ctx, tgt = split_context_target(holdout, n_context=64, seed=0)

pred  = cnp_trial_predictive(cnp, ctx, tgt, n_mc_samples=200)
y_raw = tgt.labels.mean(axis=1).astype(float)

coverage = plot_coverage_test(
    y_raw=y_raw,
    y_predicted=pred["y_cnp"],
    sigma_predicted=pred["sigma_total"],
    out_path="results/plots/cnp_only_coverage.png",
    title="CNP-only coverage on held-out HF",
)
print(coverage)  # {"1sigma": 0.68, "2sigma": 0.96, "3sigma": 1.00}
```

## Output — `cnp_trial_predictive` dict

Per-trial predictive distribution from a trained CNP. All values are 1-D arrays of length `B` (number of trials):

| Key | Meaning |
|---|---|
| `y_cnp` | Posterior mean of `y = m/N` on the trial |
| `sigma_total` | √(σ²_epistemic + σ²_aleatoric) |
| `sigma_epistemic` | CNP-decoder uncertainty (model knowledge) |
| `sigma_aleatoric` | √(p(1-p)/N) — irreducible Bernoulli noise |

## Save / load + revisit results

| Artifact | Save | Load |
|---|---|---|
| Trained CNP | `save_checkpoint(path, cnp, encoder_config, dim_theta, dim_phi, history, metadata)` | `load_checkpoint(path) -> (cnp, payload)` |
| Fitted MFGP | `save_mfgp(path, mfgp)` | `load_mfgp(path) -> MultiFidelityGP` |
| Plots | matplotlib PNGs at user-chosen `out_path` | open with any image viewer |
| Training history | included in CNP checkpoint payload (`payload["history"]`) | read after `load_checkpoint` |

```python
# Reload and revisualize a saved MFGP without re-fitting:
from core import load_mfgp, load_checkpoint
from viz import plot_coverage_test

cnp_reloaded, payload = load_checkpoint("results/cnp.ckpt")
mfgp_reloaded         = load_mfgp("results/mfgp.pkl")

# Run a fresh held-out coverage check using the reloaded models.
result = evaluate_mfgp_coverage_from_batch(
    mfgp_reloaded, cnp_reloaded, fresh_holdout_batch,
)
plot_coverage_test(
    y_raw=result["y_obs"],
    y_predicted=result["mu"],
    sigma_predicted=result["sigma"],
    out_path="results/plots/coverage_replay.png",
    title="MFGP coverage (replay)",
)
```

## Where results go (recommended layout)

The package itself does not impose an output directory — you pass paths
to every save / plot call. We recommend a flat `results/` tree:

```
results/
├── checkpoints/cnp_<run-tag>.ckpt
├── mfgp/mfgp_<run-tag>.pkl
└── plots/
    ├── cnp_only_coverage.png
    ├── mfgp_coverage.png
    └── ...
```

The `viz_output/phaseN_*/` tree is only used by the demonstration
scripts in `scripts/` — it's gitignored.
