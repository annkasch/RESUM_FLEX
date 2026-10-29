"""Validated configuration for interchangeable event surrogates.

The historical Gaussian CNP pipeline keeps its existing Config. This schema
covers the single-logit models and BDT without silently reinterpreting settings.
"""

import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from schemas.bdt import BDTConfig
from schemas.config import EncoderConfig, StrictConfigModel
from schemas.projections import ProjectionConfig as MFGPProjectionConfig
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


class LegacyCNPModel(CNPModel):
    kind: Literal["legacy_cnp"] = "legacy_cnp"


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
    CNPModel | LegacyCNPModel | MLPModel | TransformerModel | BDTModel, Field(discriminator="kind")
]


class MixupConfig(StrictConfigModel):
    alpha: float = Field(default=0.2, gt=0, allow_inf_nan=False)
    mix_context: bool = False


class SamplingConfig(StrictConfigModel):
    strategy: Literal["natural", "positive_quota", "class_aware_mixup"] = "natural"
    positive_fraction: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)

    mixup: MixupConfig | None = None

    @model_validator(mode="after")
    def quota(self):
        if (self.strategy == "positive_quota") != (self.positive_fraction is not None):
            raise ValueError("positive_fraction is required only for positive_quota sampling")
        if (self.strategy == "class_aware_mixup") != (self.mixup is not None):
            raise ValueError("mixup settings are required only for class_aware_mixup sampling")
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


class SingleObjective(StrictConfigModel):
    strategy: Literal["single"] = "single"


class RealPlusMixupObjective(StrictConfigModel):
    strategy: Literal["real_plus_mixup"] = "real_plus_mixup"
    mixup_loss_weight: float = Field(default=0.001, ge=0, allow_inf_nan=False)
    real_target_ratio: float = Field(default=1.0, gt=0, allow_inf_nan=False)


ObjectiveSpec = Annotated[SingleObjective | RealPlusMixupObjective, Field(discriminator="strategy")]


class LastLayerFineTuning(StrictConfigModel):
    """A second stage using natural real events and unweighted Bernoulli loss."""

    n_steps: int = Field(default=2000, gt=0)
    learning_rate: float = Field(default=0.0003, gt=0, allow_inf_nan=False)
    eval_every: int = Field(default=250, gt=0)
    n_events: int | None = Field(default=None, ge=3)


class RandomContext(StrictConfigModel):
    """Context size drawn uniformly from [n_min, cap] once per step.

    The cap is floor(max_fraction * n_events); max_fraction=null removes it (up to
    n_events - 2 for disjoint targets, n_events - 1 for joint). n_max is an absolute cap
    kept for old n_context_min/n_context_max configs; it replaces max_fraction.
    """

    size: Literal["random"] = "random"
    n_min: int = Field(default=16, gt=0)
    max_fraction: float | None = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    n_max: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def absolute_cap(cls, data):
        if isinstance(data, dict) and data.get("n_max") is not None:
            if data.get("max_fraction") is not None:
                raise ValueError("Set either context.max_fraction or context.n_max, not both")
            data = {**data, "max_fraction": None}
        return data


class FixedContext(StrictConfigModel):
    """The same context size n in every step."""

    size: Literal["fixed"] = "fixed"
    n: int = Field(gt=0)


ContextSpec = Annotated[RandomContext | FixedContext, Field(discriminator="size")]


class NeuralTraining(StrictConfigModel):
    backend: Literal["neural"] = "neural"
    # Run length: either n_steps, or n_passes over the training events (one pass = as
    # many target events as there are training events). Default: 10000 steps.
    n_steps: int | None = Field(default=None, gt=0)
    n_passes: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    eval_every: int = Field(default=250, gt=0)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=16, gt=0)  # voxels per step
    n_events: int = Field(default=128, ge=3)  # events drawn per voxel per step
    # disjoint: targets are the non-context events; joint: targets also include the
    # context events (Garnelo et al. 2018). Evaluation is always disjoint.
    targets: Literal["disjoint", "joint"] = "disjoint"
    context: ContextSpec = Field(default_factory=RandomContext)
    seed: int = 0
    device: Literal["cpu", "cuda"] = "cpu"
    grad_clip: float | None = Field(default=1.0, gt=0, allow_inf_nan=False)
    loss: Literal["bernoulli", "theory-truth", "practice-truth"] = "bernoulli"
    focal_gamma: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    objective: ObjectiveSpec = Field(default_factory=SingleObjective)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)
    fine_tuning: LastLayerFineTuning | None = None

    @model_validator(mode="before")
    @classmethod
    def run_length(cls, data):
        if isinstance(data, dict) and data.get("n_passes") is None and data.get("n_steps") is None:
            data = {**data, "n_steps": 10000}
        return data

    @model_validator(mode="before")
    @classmethod
    def legacy_context(cls, data):
        """Map old n_context_min/n_context_max (absolute, defaults 16/64) to context.

        A context block without size means size=random.
        """
        if isinstance(data, dict) and isinstance(data.get("context"), dict):
            data = {**data, "context": {"size": "random", **data["context"]}}
        if isinstance(data, dict) and {"n_context_min", "n_context_max"} & data.keys():
            if "context" in data:
                raise ValueError("Use context or n_context_min/n_context_max, not both")
            data = dict(data)
            n_min, n_max = data.pop("n_context_min", 16), data.pop("n_context_max", 64)
            data["context"] = {"size": "random", "n_min": n_min, "n_max": n_max}
        return data

    def context_range(self, n_events: int | None = None) -> tuple[int, int]:
        """Smallest and largest context size for episodes of n_events per voxel."""
        n = self.n_events if n_events is None else n_events
        limit = n - 1 if self.targets == "joint" else n - 2
        context = self.context
        if context.size == "fixed":
            low = high = context.n
        else:
            low = context.n_min
            if context.n_max is not None:
                high = context.n_max
            elif context.max_fraction is not None:
                high = int(context.max_fraction * n + 1e-9)  # floor, robust to rounding
            else:
                high = limit
        if not 1 <= low <= high <= limit:
            need = "one non-context target" if self.targets == "joint" else "two target events"
            raise ValueError(
                f"Context sizes [{low}, {high}] must satisfy 1 <= min <= max <= {limit} "
                f"for n_events={n} (at least {need})"
            )
        return low, high

    def target_events_per_step(self) -> float:
        """Expected number of scored target events in one step."""
        if self.targets == "joint":
            return float(self.batch_size * self.n_events)
        low, high = self.context_range()
        return self.batch_size * (self.n_events - (low + high) / 2)

    def resolve_n_steps(self, training_events: int) -> int:
        """n_steps, or the steps needed for n_passes over training_events target events."""
        if self.n_steps is not None:
            return self.n_steps
        return max(1, math.ceil(self.n_passes * training_events / self.target_events_per_step()))

    @model_validator(mode="after")
    def compatible(self):
        if self.n_steps is not None and self.n_passes is not None:
            raise ValueError("Set either n_steps or n_passes, not both")
        self.context_range()
        if self.targets == "joint" and self.sampling.strategy != "natural":
            raise ValueError("targets=joint currently requires natural sampling")
        if self.sampling.strategy == "class_aware_mixup" and self.weighting.strategy != "none":
            raise ValueError("class_aware_mixup currently requires weighting=none")
        if self.objective.strategy == "real_plus_mixup":
            if self.sampling.strategy != "class_aware_mixup":
                raise ValueError("real_plus_mixup requires class_aware_mixup sampling")
            if self.focal_gamma != 0 or self.weighting.strategy != "none":
                raise ValueError("real_plus_mixup requires an unfocused loss without extra weights")
        if self.loss != "bernoulli" and self.focal_gamma != 0:
            raise ValueError("Legacy losses require focal_gamma=0")
        quota = self.sampling.strategy == "positive_quota"
        corrected = self.weighting.strategy == "sampling_correction"
        if quota != corrected:
            raise ValueError("positive_quota and sampling_correction must be used together")
        if self.fine_tuning is not None:
            try:
                self.context_range(self.fine_tuning.n_events or self.n_events)
            except ValueError as error:
                raise ValueError(f"Fine-tuning events: {error}") from None
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


class TrainingStage(StrictConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    start_from: str = "initial"
    trainable: Literal["all", "output_layer"] = "all"
    training: TrainingSpec
    selection: SelectionMetric = "voxel_rate_mae"

    @model_validator(mode="after")
    def no_nested_schedule(self):
        if self.training.backend == "neural" and self.training.fine_tuning is not None:
            raise ValueError("Use separate stages, not nested fine_tuning")
        if self.training.backend == "bdt" and self.trainable != "all":
            raise ValueError("BDT does not support output-layer fine-tuning")
        return self


class SurrogateConfig(StrictConfigModel):
    model: ModelSpec
    training: TrainingSpec
    selection: SelectionMetric = "voxel_rate_mae"

    @model_validator(mode="after")
    def backend_matches(self):
        if (self.model.kind == "bdt") != (self.training.backend == "bdt"):
            raise ValueError("BDT requires backend=bdt; neural models require backend=neural")
        if self.training.backend == "neural":
            legacy = self.model.kind == "legacy_cnp"
            if legacy != (self.training.loss in ("theory-truth", "practice-truth")):
                raise ValueError(
                    "legacy_cnp requires an explicit theory-truth or practice-truth loss; "
                    "single-logit models require bernoulli"
                )
        return self


Fidelity = Literal["lf", "hf"]


class ModelMeanLevel(StrictConfigModel):
    """GP level from an event model: mean predicted p per voxel of a prepared partition.

    ``inference``: a file written by ``python -m core.surrogates predict``; without it the
    run's (or the configured) checkpoint predicts at fit time. Context models need labeled
    voxels for their context; context-free models also predict unlabeled voxels.
    """

    source: Literal["model_mean"]
    partition: str
    fidelity: Fidelity = "lf"
    inference: Path | None = None
    max_voxels: int | None = Field(default=None, gt=0)


class ObservedFractionLevel(StrictConfigModel):
    """GP level from simulation: observed hit fraction m/N over all events of each voxel."""

    source: Literal["observed_fraction"]
    partition: str
    fidelity: Fidelity = "hf"
    max_voxels: int | None = Field(default=None, gt=0)


GPLevel = Annotated[ModelMeanLevel | ObservedFractionLevel, Field(discriminator="source")]


# Held-out test partitions: never GP training levels, evaluated only with unblind_test.
TEST_PARTITIONS = ("test", "cnp_test", "gp_test")


class GPEvaluation(StrictConfigModel):
    """Held-out voxels whose observed fraction is compared with one GP level (default: top)."""

    partition: str
    fidelity: Fidelity = "hf"
    level: int | None = None


class MFGPStageConfig(StrictConfigModel):
    # Ordered from the lowest to the highest fidelity; the last level is the target.
    levels: list[GPLevel] = Field(min_length=2)
    evaluation: list[GPEvaluation] = Field(default_factory=list)
    projections: MFGPProjectionConfig = Field(default_factory=MFGPProjectionConfig)
    output_transform: Literal["identity", "log"] = "identity"
    # Log mode: observed fractions become (m + a) / (N + 2a); model means are floored.
    raw_pseudocount: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    cnp_log_floor: float = Field(default=1e-12, gt=0, lt=1, allow_inf_nan=False)
    kernel: Literal["rbf", "matern52"] = "rbf"
    n_restarts: int = Field(default=5, gt=0)
    seed: int = 0
    # Context events per voxel when a context model computes a model_mean level.
    n_context: int = Field(default=64, gt=0)
    # Test partitions stay blinded until all choices are final; then set true once.
    unblind_test: bool = False

    @model_validator(mode="after")
    def consistent_levels(self):
        limits = {}
        for level in self.levels:
            if level.partition in TEST_PARTITIONS:
                raise ValueError(f"Test partition {level.partition} can't be a GP training level")
        for evaluation in self.evaluation:
            if evaluation.partition in TEST_PARTITIONS and not self.unblind_test:
                raise ValueError(
                    f"{evaluation.partition} is blinded; set unblind_test: true once all "
                    "choices are final"
                )
        for level in self.levels:
            key = level.partition, level.fidelity
            if limits.setdefault(key, level.max_voxels) != level.max_voxels:
                raise ValueError(f"Levels on {key[0]}/{key[1]} need the same max_voxels")
        for evaluation in self.evaluation:
            if evaluation.level is not None and not 0 <= evaluation.level < len(self.levels):
                raise ValueError(f"evaluation level must be in [0, {len(self.levels) - 1}]")
        return self


class MFGPRunConfig(MFGPStageConfig):
    """Standalone MFGP fit on prepared data, independent of event-model training."""

    data_directory: Path
    # Optional data-preparation YAML; prepared data is regenerated when it differs.
    data_config: Path | None = None
    # Event model for model_mean levels without an inference file.
    checkpoint: Path | None = None
    output_directory: Path

    @model_validator(mode="after")
    def model_available(self):
        if self.checkpoint is None and any(
            level.source == "model_mean" and level.inference is None for level in self.levels
        ):
            raise ValueError("model_mean levels without an inference file need a checkpoint")
        return self


def load_mfgp_config(path: str | Path) -> MFGPRunConfig:
    """Standalone MFGP config; paths resolve relative to the YAML file."""
    path = Path(path).resolve()
    config = MFGPRunConfig.model_validate(yaml.safe_load(path.read_text()))

    def resolve(value):
        return value if value is None or value.is_absolute() else (path.parent / value).resolve()

    for name in ("data_directory", "data_config", "checkpoint", "output_directory"):
        setattr(config, name, resolve(getattr(config, name)))
    for level in config.levels:
        if level.source == "model_mean":
            level.inference = resolve(level.inference)
    return config


class SurrogateRunConfig(SurrogateConfig):
    lf_validation: bool = True
    hf_validation: bool = True
    # Test sets that exist; they are evaluated only with unblind_test (blinded by default,
    # so model choices are made on validation only).
    test_fidelities: list[Literal["lf", "hf"]] = Field(default_factory=list)
    unblind_test: bool = False
    stages: list[TrainingStage] | None = Field(default=None, min_length=1)
    initial_checkpoint: Path | None = None
    mfgp_checkpoint: str = "best"
    mfgp: MFGPStageConfig | None = None
    data_directory: Path
    # Optional data-preparation YAML. When set, each run first compares it with the settings
    # stored in data_directory and regenerates the prepared data if they differ.
    data_config: Path | None = None
    output_directory: Path
    validation_context_events: int = Field(default=64, gt=0)
    # Extra context sizes evaluated after training (read-out only, not selection). The
    # targets are the events after the largest context size, so every size scores the
    # same targets; the contexts are nested.
    validation_context_sweep: list[int] = Field(default_factory=list)
    validation_seed: int = 12345
    # Checkpoint selection during training uses a seeded subset of LF validation voxels;
    # None uses all. Full validation and test sets are evaluated once after training.
    selection_max_voxels: int | None = Field(default=None, gt=0)
    # CPU threads for evaluation passes only; training steps keep their own thread count.
    evaluation_threads: int = Field(default=1, gt=0)

    @property
    def validation_context_pool(self) -> int:
        """Events per voxel reserved for contexts; the targets are all later events."""
        return max([self.validation_context_events, *self.validation_context_sweep])

    @model_validator(mode="after")
    def valid_context_sweep(self):
        sweep = self.validation_context_sweep
        if any(n <= 0 for n in sweep) or len(set(sweep)) != len(sweep):
            raise ValueError("validation_context_sweep needs unique positive sizes")
        return self

    @model_validator(mode="after")
    def unique_tests(self):
        if len(set(self.test_fidelities)) != len(self.test_fidelities):
            raise ValueError("test_fidelities must be unique")
        if self.stages:
            if self.training.backend == "neural" and self.training.fine_tuning is not None:
                raise ValueError("Specify stages or fine_tuning, not both")
            seen = set()
            for stage in self.stages:
                if stage.name in seen or stage.name in {"best", "final", "initial"}:
                    raise ValueError("Stage names must be unique and not reserved")
                SurrogateConfig(model=self.model, training=stage.training, selection=stage.selection)
                if stage.start_from != "initial":
                    parts = stage.start_from.split("/")
                    if len(parts) != 2 or parts[0] not in seen or parts[1] not in {"best", "final"}:
                        raise ValueError("Stage start_from must reference an earlier stage/best or final")
                if stage.training.backend == "bdt" and stage.start_from != "initial":
                    raise ValueError("BDT continuation is not supported")
                seen.add(stage.name)
        choices = {"best", "final", "pretraining"}
        choices.update(f"stages/{s.name}/{which}" for s in self.stages or []
                       for which in ("best", "final"))
        if self.mfgp_checkpoint not in choices:
            raise ValueError("Unknown mfgp_checkpoint")
        return self


def load_surrogate_config(path: str | Path) -> SurrogateRunConfig:
    """Resolve data/output paths relative to the YAML file, not the caller's cwd."""
    path = Path(path).resolve()
    config = SurrogateRunConfig.model_validate(yaml.safe_load(path.read_text()))
    for name in ("data_directory", "data_config", "output_directory", "initial_checkpoint"):
        value = getattr(config, name)
        if value is not None and not value.is_absolute():
            setattr(config, name, (path.parent / value).resolve())
    for level in config.mfgp.levels if config.mfgp else []:
        if level.source == "model_mean" and level.inference and not level.inference.is_absolute():
            level.inference = (path.parent / level.inference).resolve()
    return config
