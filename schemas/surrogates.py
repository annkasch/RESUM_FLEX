"""Validated configuration for interchangeable event surrogates.

The historical Gaussian CNP pipeline keeps its existing Config. This schema
covers the single-logit models and BDT without silently reinterpreting settings.
"""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from schemas.bdt import BDTConfig
from schemas.config import EncoderConfig, StrictConfigModel
from schemas.tabular_transformer import TabularTransformerConfig


def default_encoder():
    return EncoderConfig(type="mlp", latent_dim=64, hidden_dims=[128, 128])


class CNPModel(StrictConfigModel):
    kind: Literal["cnp"] = "cnp"
    encoder: EncoderConfig = Field(default_factory=default_encoder)

    @model_validator(mode="after")
    def supported_encoder(self):
        if self.encoder.type != "mlp":
            raise ValueError(
                "This CNP supports the MLP encoder; use kind=transformer for tabular attention"
            )
        return self


class MLPArchitecture(StrictConfigModel):
    hidden_dims: list[Annotated[int, Field(gt=0)]] = Field(
        default_factory=lambda: [128, 128], min_length=1
    )
    dropout: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)


class MLPModel(StrictConfigModel):
    kind: Literal["mlp"] = "mlp"
    architecture: MLPArchitecture = Field(default_factory=MLPArchitecture)


class TransformerModel(StrictConfigModel):
    kind: Literal["transformer"] = "transformer"
    architecture: TabularTransformerConfig = Field(
        default_factory=lambda: TabularTransformerConfig(
            architecture="ft_transformer",
            token_dim=64,
            n_heads=4,
            n_layers=2,
            feedforward_dim=224,
            inference_chunk_size=512,
        )
    )


class BDTModel(StrictConfigModel):
    kind: Literal["bdt"] = "bdt"
    architecture: BDTConfig = Field(default_factory=BDTConfig)


ModelSpec = Annotated[
    CNPModel | MLPModel | TransformerModel | BDTModel, Field(discriminator="kind")
]


class SamplingConfig(StrictConfigModel):
    strategy: Literal["natural", "positive_quota"] = "natural"
    positive_fraction: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def quota(self):
        if (self.strategy == "positive_quota") != (self.positive_fraction is not None):
            raise ValueError("positive_fraction is required only for positive_quota sampling")
        return self


class WeightingConfig(StrictConfigModel):
    strategy: Literal["none", "sampling_correction", "class_weights"] = "none"
    positive: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    negative: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def weights(self):
        if self.strategy != "class_weights" and (self.positive != 1 or self.negative != 1):
            raise ValueError("Explicit weights require strategy=class_weights")
        return self


class NeuralTraining(StrictConfigModel):
    backend: Literal["neural"] = "neural"
    n_steps: int = Field(default=10000, gt=0)
    eval_every: int = Field(default=250, gt=0)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=16, gt=0)
    n_events: int = Field(default=128, ge=3)
    n_context_min: int = Field(default=16, gt=0)
    n_context_max: int = Field(default=64, gt=0)
    seed: int = 0
    device: Literal["cpu", "cuda"] = "cpu"
    grad_clip: float | None = Field(default=1.0, gt=0, allow_inf_nan=False)
    focal_gamma: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)

    @model_validator(mode="after")
    def compatible(self):
        if not self.n_context_min <= self.n_context_max <= self.n_events - 2:
            raise ValueError("Context range must leave at least two target events")
        quota = self.sampling.strategy == "positive_quota"
        corrected = self.weighting.strategy == "sampling_correction"
        if quota != corrected:
            raise ValueError("positive_quota and sampling_correction must be used together")
        return self


class TreeTraining(StrictConfigModel):
    backend: Literal["bdt"] = "bdt"
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)

    @model_validator(mode="after")
    def compatible(self):
        if self.weighting.strategy == "sampling_correction":
            raise ValueError("BDT fits all natural events; sampling_correction is not applicable")
        return self


TrainingSpec = Annotated[NeuralTraining | TreeTraining, Field(discriminator="backend")]
SelectionMetric = Literal["voxel_rate_mae", "average_precision", "bernoulli_log_loss"]


class SurrogateConfig(StrictConfigModel):
    model: ModelSpec
    training: TrainingSpec
    selection: SelectionMetric = "voxel_rate_mae"

    @model_validator(mode="after")
    def backend_matches(self):
        if (self.model.kind == "bdt") != (self.training.backend == "bdt"):
            raise ValueError("BDT requires backend=bdt; neural models require backend=neural")
        return self


class SurrogateRunConfig(SurrogateConfig):
    data_directory: Path
    output_directory: Path
    validation_context_events: int = Field(default=64, gt=0)
    validation_seed: int = 12345


def load_surrogate_config(path: str | Path) -> SurrogateRunConfig:
    """Resolve data/output paths relative to the YAML file, not the caller's cwd."""
    path = Path(path).resolve()
    config = SurrogateRunConfig.model_validate(yaml.safe_load(path.read_text()))
    for name in ("data_directory", "output_directory"):
        value = getattr(config, name)
        if not value.is_absolute():
            setattr(config, name, (path.parent / value).resolve())
    return config
