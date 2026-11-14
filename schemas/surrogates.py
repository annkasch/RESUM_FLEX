"""Validated configuration for interchangeable event surrogates.

The historical Gaussian CNP pipeline keeps its existing Config. This schema
covers the single-logit models and BDT without silently reinterpreting settings.
"""

import math
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AliasChoices, Field, field_validator, model_validator

from schemas.bdt import BDTConfig
from schemas.config import EncoderConfig, StrictConfigModel
from schemas.projections import ProjectionConfig as MFGPProjectionConfig
from schemas.tabular_transformer import TabularTransformerConfig


def default_encoder():
    return EncoderConfig(type="mlp", latent_dim=64, hidden_dims=[128, 128])


class BayesianLastLayer(StrictConfigModel):
    """Laplace posterior over the final linear layer, fitted after training.

    The trained weights are the posterior mean. The precision is the exact Bernoulli
    Hessian over all natural training events plus an isotropic Gaussian prior:
    ``prior_precision`` = "evidence" maximizes the Laplace marginal likelihood (training
    data only), a number fixes it. ``draws`` weight samples give event and design point
    predictive distributions; ``seed`` makes them reproducible.
    """

    prior_precision: Literal["evidence"] | Annotated[float, Field(gt=0, allow_inf_nan=False)] = (
        "evidence"
    )
    draws: int = Field(default=1000, ge=100)
    seed: int = 0


class CNPModel(StrictConfigModel):
    kind: Literal["cnp"] = "cnp"
    encoder: EncoderConfig = Field(default_factory=default_encoder)
    last_layer: BayesianLastLayer | None = None

    @model_validator(mode="after")
    def supported_encoder(self):
        if self.encoder.type != "mlp":
            raise ValueError(
                "This CNP supports the MLP encoder; use kind=transformer for tabular attention"
            )
        return self


class LegacyCNPModel(StrictConfigModel):
    kind: Literal["legacy_cnp"] = "legacy_cnp"
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
    last_layer: BayesianLastLayer | None = None


class FieldPosition(StrictConfigModel):
    """Event columns that, added to theta in physical units, give one continuous position.

    For example theta = a position and phi[0:3] = an emission offset from it. The sum uses the
    data normalization stored with the model, so it is exact in physical units; the field never
    sees the design point structure.
    """

    phi_columns: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)


class FieldDirection(StrictConfigModel):
    """Event columns of a 3-vector direction (any length).

    ``harmonics``: real spherical harmonics up to ``degree`` of the unit vector (the length is
    dropped). ``momentum``: the vector itself in the normalized units of the training data,
    length included (for events: the wavelength); ``degree`` is unused.
    """

    phi_columns: list[Annotated[int, Field(ge=0)]] = Field(min_length=3, max_length=3)
    encoding: Literal["harmonics", "momentum"] = "harmonics"
    degree: int = Field(default=4, ge=0, le=8)


class FourierEncoding(StrictConfigModel):
    """sin/cos of each position coordinate at ``n_frequencies`` octave-spaced frequencies
    up to ``max_frequency`` (cycles per unit of normalized theta), plus the raw value.

    The finest resolvable half-period is 1 / (2 max_frequency) normalized units.
    """

    kind: Literal["fourier"] = "fourier"
    n_frequencies: int = Field(default=8, ge=1, le=16)
    max_frequency: float = Field(default=8.0, gt=0, allow_inf_nan=False)


class RawPosition(StrictConfigModel):
    """The position in normalized units, unencoded (an MLP on raw coordinates)."""

    kind: Literal["none"] = "none"


class FieldModel(StrictConfigModel):
    """Neural field for point-cloud data: a network on encoded event coordinates,
    p(position, direction, rest). The direction is a 3-vector (spherical harmonics).

    Position = theta (+ ``position.phi_columns`` in physical units) through the Fourier
    encoding; ``direction`` columns through spherical harmonics; all other phi columns
    enter unencoded. Context-free like the MLP. ``architecture``: an MLP (``hidden_dims``),
    or a tabular transformer with one token per encoded column (``token_dim``, ...; the
    stream transformer).
    """

    kind: Literal["field"] = "field"
    # Untagged: both forbid extra fields, so the keys decide (MLP by default).
    architecture: MLPArchitecture | TabularTransformerConfig = Field(
        default_factory=MLPArchitecture
    )
    position: FieldPosition | None = None
    direction: FieldDirection | None = None
    # Untagged: configs without ``kind`` stay Fourier (both models forbid extra fields).
    encoding: FourierEncoding | RawPosition = Field(default_factory=FourierEncoding)
    last_layer: BayesianLastLayer | None = None

    @model_validator(mode="after")
    def distinct_columns(self):
        groups = [g.phi_columns for g in (self.position, self.direction) if g is not None]
        columns = [c for group in groups for c in group]
        if len(set(columns)) != len(columns):
            raise ValueError("position and direction phi_columns must be distinct")
        return self


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
    last_layer: BayesianLastLayer | None = None


class BDTModel(StrictConfigModel):
    kind: Literal["bdt"] = "bdt"
    architecture: BDTConfig = Field(default_factory=BDTConfig)


ModelSpec = Annotated[
    CNPModel | LegacyCNPModel | MLPModel | FieldModel | TransformerModel | BDTModel,
    Field(discriminator="kind"),
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


# Event-model kinds that condition on labelled context events of the same design point.
CONTEXT_KINDS = frozenset({"cnp", "legacy_cnp"})


def uses_context(kind: str) -> bool:
    """Whether an event model of this kind conditions on context events."""
    return kind in CONTEXT_KINDS


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
    batch_size: int = Field(default=16, gt=0)  # design points per step
    n_events: int = Field(default=128, ge=3)  # events drawn per design point per step
    # Context models only (cnp, legacy_cnp); context-free models score all n_events events.
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

    def context_range(self, n_events: int | None = None, *, context=True) -> tuple[int, int]:
        """Smallest and largest context size for episodes of n_events per design point.

        context=False (context-free models): (0, 0), every drawn event is a target.
        """
        if not context:
            return 0, 0
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

    def target_events_per_step(self, *, context=True) -> float:
        """Expected number of scored target events in one step (context: context model)."""
        if self.targets == "joint" or not context:
            return float(self.batch_size * self.n_events)
        low, high = self.context_range()
        return self.batch_size * (self.n_events - (low + high) / 2)

    def resolve_n_steps(self, training_events: int, *, context=True) -> int:
        """n_steps, or the steps needed for n_passes over training_events target events."""
        if self.n_steps is not None:
            return self.n_steps
        per_step = self.target_events_per_step(context=context)
        return max(1, math.ceil(self.n_passes * training_events / per_step))

    def check_context(self, kind: str):
        """Context sizes fit the episodes of a context model; context-free models keep the
        defaults (their context and targets settings would be inert)."""
        if uses_context(kind):
            self.context_range()
            if self.fine_tuning is not None:
                try:
                    self.context_range(self.fine_tuning.n_events or self.n_events)
                except ValueError as error:
                    raise ValueError(f"Fine-tuning events: {error}") from None
        elif (
            self.targets != "disjoint"
            or self.context != RandomContext()
            or (self.sampling.mixup is not None and self.sampling.mixup.mix_context)
        ):
            raise ValueError(
                f"{kind} is context-free and scores all n_events events per design point; "
                "remove training.context, targets, n_context_min/n_context_max and "
                "sampling.mixup.mix_context"
            )

    @model_validator(mode="after")
    def compatible(self):
        if self.n_steps is not None and self.n_passes is not None:
            raise ValueError("Set either n_steps or n_passes, not both")
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
# design_point_count_nll: mean -log Binomial(k | n, mean predicted p) per design point
# (plug-in); far more sensitive than the event log loss when design points have few hits.
SelectionMetric = Literal[
    "design_point_rate_mae", "average_precision", "bernoulli_log_loss", "design_point_count_nll"
]


class TrainingStage(StrictConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    start_from: str = "initial"
    trainable: Literal["all", "output_layer"] = "all"
    training: TrainingSpec
    selection: SelectionMetric = "design_point_rate_mae"

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
    selection: SelectionMetric = "design_point_rate_mae"

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
            self.training.check_context(self.model.kind)
        return self


Fidelity = Literal["lf", "hf"]


class ModelMeanLevel(StrictConfigModel):
    """GP level from an event model: mean predicted p per design point of a prepared partition.

    ``inference``: a file written by ``python -m core.surrogates predict``; without it the
    run's (or the configured) checkpoint predicts at fit time. Context models need labeled
    design points for their context; context-free models also predict unlabeled design points.

    ``noise``: ``learned`` (the level's constant noise only) or ``last_layer``: each point also
    gets the variance of its mean p over the event model's last-layer draws (log mode: of
    log p) as a fixed noise on top of the learned level noise. Needs a last-layer posterior
    (checkpoint) or an inference file with ``probability_draws``.
    """

    source: Literal["model_mean"]
    partition: str
    fidelity: Fidelity = "lf"
    inference: Path | None = None
    noise: Literal["learned", "last_layer"] = "learned"
    max_design_points: int | None = Field(default=None, gt=0)


class ObservedFractionLevel(StrictConfigModel):
    """GP level from simulation: observed hit fraction m/N over all events of each design point."""

    source: Literal["observed_fraction"]
    partition: str
    fidelity: Fidelity = "hf"
    max_design_points: int | None = Field(default=None, gt=0)


GPLevel = Annotated[ModelMeanLevel | ObservedFractionLevel, Field(discriminator="source")]


# Held-out test partitions: never GP training levels, evaluated only with unblind_test.
TEST_PARTITIONS = ("test", "cnp_test", "gp_test")


class GPEvaluation(StrictConfigModel):
    """Held-out design points whose observed fraction is compared with one GP level.

    Default level: the top one.
    """

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
    # Refits at the rate the previous fit predicts: log mode corrects observed-fraction
    # targets so they average to log p (pseudocount + Jensen offset); binomial noise is
    # updated. Stop when nothing changes by > 0.01, at most this often. None: 8 when there
    # is something to refit (log mode or binomial noise), else 0; 0 turns the log
    # correction off. Identity mode with learned noise: only 0.
    max_refits: int | None = Field(default=None, ge=0)
    # learned: one constant noise variance per level. binomial: observed-fraction points
    # also get their count noise at the predicted rate (log: Var[log y], identity:
    # p(1 - p)/N), held-out points at their own N, on top of the learned level noise.
    observation_noise: Literal["learned", "binomial"] = "learned"
    kernel: Literal["rbf", "matern52"] = "rbf"
    n_restarts: int = Field(default=5, gt=0)
    seed: int = 0
    # Fixed y range of the coverage plots (observed fraction); None = from the data.
    coverage_ylim: tuple[float, float] | None = None

    @field_validator("coverage_ylim")
    @classmethod
    def _ordered_coverage_ylim(cls, value):
        if value is not None and not value[0] < value[1]:
            raise ValueError(f"coverage_ylim must be (low, high) with low < high, got {value}")
        return value

    # Context events per design point when a context model computes a model_mean level.
    n_context: int = Field(default=64, gt=0)
    # Test partitions stay blinded until all choices are final; then set true once.
    unblind_test: bool = False

    @model_validator(mode="after")
    def consistent_levels(self):
        binomial = self.observation_noise == "binomial"
        if self.max_refits and self.output_transform != "log" and not binomial:
            raise ValueError(
                "max_refits needs output_transform: log or observation_noise: binomial"
            )
        if binomial and self.max_refits == 0:
            raise ValueError("observation_noise: binomial needs max_refits > 0")
        if binomial and not any(level.source == "observed_fraction" for level in self.levels):
            raise ValueError("observation_noise: binomial needs an observed_fraction level")
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
            if limits.setdefault(key, level.max_design_points) != level.max_design_points:
                raise ValueError(f"Levels on {key[0]}/{key[1]} need the same max_design_points")
        for evaluation in self.evaluation:
            if evaluation.level is not None and not 0 <= evaluation.level < len(self.levels):
                raise ValueError(f"evaluation level must be in [0, {len(self.levels) - 1}]")
        return self


class MFGPRunConfig(MFGPStageConfig):
    """Standalone MFGP fit on prepared data, independent of event-model training."""

    data_directory: Path
    # Dataset provider 'module:Class' (core.surrogates.datasets); None reads .npz batches.
    data_provider: str | None = None
    # Optional data-preparation YAML for the provider; data are rebuilt when outdated.
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


class CountSet(StrictConfigModel):
    """Prepared partition whose per-design-point hit counts enter the binomial likelihood."""

    partition: str
    fidelity: Fidelity
    # Fit on a seeded subset of this many design points (exact Laplace GPs scale as O(n^3)).
    max_design_points: int | None = Field(default=None, gt=0)


class CountEvaluation(StrictConfigModel):
    partition: str
    fidelity: Fidelity = "hf"


class BinomialCovariate(StrictConfigModel):
    """Event-model mean probability per design point as GP covariate z = logit(p).

    ``inference`` maps every observation and evaluation set ("partition/fidelity") to a file
    of ``python -m core.surrogates predict`` for exactly that partition (paths relative to
    the YAML file). The event model must not have been trained or selected on the labels of
    these sets, otherwise their counts enter twice.
    """

    inference: dict[str, Path] = Field(min_length=1)


class BinomialGPRunConfig(StrictConfigModel):
    """Single spatial GP on hit counts: m ~ Binomial(N, sigmoid(f(theta))), Laplace posterior.

    All observation sets share one latent probability; fidelities differ only in N, which
    enters the likelihood exactly. An event model may enter only as the optional covariate.
    """

    observations: list[CountSet] = Field(min_length=1)
    evaluation: list[CountEvaluation] = Field(default_factory=list)
    kernel: Literal["rbf", "matern52"] = "matern52"
    n_restarts: int = Field(default=2, gt=0)
    max_iters: int = Field(default=300, gt=0)
    seed: int = 42
    # Monte Carlo draws of the observation-predictive distribution per evaluation set.
    prediction_draws: int = Field(default=32768, ge=2048)
    # Optional event-model covariate: logit p = b logit(p_model) + c + delta(theta).
    covariate: BinomialCovariate | None = None
    # Fixed y range of the coverage plots (observed fraction); None = from the data.
    coverage_ylim: tuple[float, float] | None = None

    @field_validator("coverage_ylim")
    @classmethod
    def _ordered_coverage_ylim(cls, value):
        if value is not None and not value[0] < value[1]:
            raise ValueError(f"coverage_ylim must be (low, high) with low < high, got {value}")
        return value

    unblind_test: bool = False
    data_directory: Path
    data_provider: str | None = None
    data_config: Path | None = None
    output_directory: Path

    @model_validator(mode="after")
    def held_out_sets(self):
        for item in self.observations:
            if item.partition in TEST_PARTITIONS:
                raise ValueError(f"Test partition {item.partition} can't be a training set")
        for item in self.evaluation:
            if item.partition in TEST_PARTITIONS and not self.unblind_test:
                raise ValueError(
                    f"{item.partition} is blinded; set unblind_test: true once all choices "
                    "are final"
                )
        if self.covariate is not None:
            needed = {f"{i.partition}/{i.fidelity}" for i in [*self.observations, *self.evaluation]}
            missing = sorted(needed - set(self.covariate.inference))
            if missing:
                raise ValueError(f"covariate.inference has no file for {missing}")
        return self


def load_binomial_config(path: str | Path) -> BinomialGPRunConfig:
    """Binomial GP config; paths resolve relative to the YAML file."""
    path = Path(path).resolve()
    config = BinomialGPRunConfig.model_validate(yaml.safe_load(path.read_text()))
    for name in ("data_directory", "data_config", "output_directory"):
        value = getattr(config, name)
        if value is not None and not value.is_absolute():
            setattr(config, name, (path.parent / value).resolve())
    if config.covariate is not None:
        config.covariate.inference = {
            key: value if value.is_absolute() else (path.parent / value).resolve()
            for key, value in config.covariate.inference.items()
        }
    return config


class TrainingSet(StrictConfigModel):
    """One prepared partition/fidelity whose labelled events train the event model."""

    partition: str
    fidelity: Fidelity


class FeatureCalibrationConfig(StrictConfigModel):
    """Calibration per input column in the read-out and on stream validation events
    (docs/evaluation.md): equal-count bins of each column's value."""

    bins: int = Field(default=10, ge=2)


class ResidualTestConfig(StrictConfigModel):
    """Residual learnability test (read-out only, needs the ``lgbm`` extra).

    LightGBM learns a correction to the model logit from (inputs, logit p) by
    cross-fitting; its held-out log-loss gain shows structure the event model missed
    (docs/evaluation.md). Design-point sets are split by design point, stream
    validation records by event.
    """

    folds: int = Field(default=5, ge=2)
    # Below this many positives the test is not run (metrics None): precision is set by
    # positives, not by design points.
    min_positives: int = Field(default=50, ge=1)
    # Small fixed trees; the number of trees comes from inner early stopping.
    num_leaves: int = Field(default=7, ge=2)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    min_data_in_leaf: int = Field(default=200, ge=1)
    lambda_l2: float = Field(default=1.0, ge=0)
    max_trees: int = Field(default=500, ge=1)
    patience: int = Field(default=20, ge=1)
    threads: int = Field(default=1, ge=1)
    # Stream validation events only: keep a seeded uniform fraction of all events so that
    # about this many positives remain (precision is set by positives; cost by events).
    # None = all events. Not allowed in training configs (no stream records there).
    max_positives: int | None = Field(default=20_000, ge=1)


class SurrogateRunConfig(SurrogateConfig):
    lf_validation: bool = True
    hf_validation: bool = True
    # Validation fidelity that selects the checkpoint (it must be loaded). HF selection makes
    # that set's scores selection-biased; keep a blinded test set for the final comparison.
    selection_fidelity: Literal["lf", "hf"] = "lf"
    # Test sets that exist; they are evaluated only with unblind_test (blinded by default,
    # so model choices are made on validation only).
    test_fidelities: list[Literal["lf", "hf"]] = Field(default_factory=list)
    unblind_test: bool = False
    stages: list[TrainingStage] | None = Field(default=None, min_length=1)
    initial_checkpoint: Path | None = None
    mfgp_checkpoint: str = "best"
    mfgp: MFGPStageConfig | None = None
    data_directory: Path
    # Dataset provider 'module:Class' (core.surrogates.datasets); None reads .npz batches.
    data_provider: str | None = None
    # Provider partitions holding the event-model train/validation/test sets.
    partitions: dict[Literal["train", "validation", "test"], str] = Field(
        default_factory=lambda: {"train": "train", "validation": "validation", "test": "test"}
    )
    # Train on the design points of several sets instead of partitions.train at LF, e.g. the GP's
    # training data (LF2 + HF train) for a model that predicts design point rates itself. Selection
    # still uses partitions.validation; the sets may not include it or partitions.test.
    train_sets: list[TrainingSet] | None = Field(default=None, min_length=1)
    # Optional data-preparation YAML for the provider. When set, each run first lets the
    # provider rebuild data_directory if its settings or raw inputs changed.
    data_config: Path | None = None
    output_directory: Path
    validation_context_events: int = Field(default=64, gt=0)
    # Extra context sizes evaluated after training (read-out only, not selection). The
    # targets are the events after the largest context size, so every size scores the
    # same targets; the contexts are nested.
    validation_context_sweep: list[int] = Field(default_factory=list)
    validation_seed: int = 12345
    # Checkpoint selection during training uses a seeded subset of LF validation design points;
    # None uses all. Full validation and test sets are evaluated once after training.
    selection_max_design_points: int | None = Field(default=None, gt=0)
    # CPU threads for evaluation passes only; training steps keep their own thread count.
    evaluation_threads: int = Field(default=1, gt=0)
    # Label draws per design point for the within-design-point null (read-out); the ±3σ
    # coverage rests on ~draws/740 draws per tail (1000: rough, 10 000: ~10× the cost).
    within_point_null_draws: int = Field(default=1000, ge=100)
    # Calibration per input column on every held-out split (read-out).
    feature_calibration: FeatureCalibrationConfig = Field(default_factory=FeatureCalibrationConfig)
    # Residual learnability test on every held-out split (read-out); None = off.
    residual_test: ResidualTestConfig | None = None

    @property
    def validation_context_pool(self) -> int:
        """Events per design point reserved for contexts; the targets are all later events."""
        return max([self.validation_context_events, *self.validation_context_sweep])

    @model_validator(mode="after")
    def valid_context_sweep(self):
        sweep = self.validation_context_sweep
        if any(n <= 0 for n in sweep) or len(set(sweep)) != len(sweep):
            raise ValueError("validation_context_sweep needs unique positive sizes")
        return self

    @model_validator(mode="after")
    def residual_test_on_design_points(self):
        # A changed value would be inert here; the default is accepted (resolved configs).
        default = ResidualTestConfig.model_fields["max_positives"].default
        if self.residual_test is not None and self.residual_test.max_positives != default:
            raise ValueError("residual_test.max_positives is for stream validation events only")
        return self

    @model_validator(mode="after")
    def selection_set_loaded(self):
        if self.selection_fidelity == "hf" and not self.hf_validation:
            raise ValueError("selection_fidelity: hf needs hf_validation: true")
        return self

    @model_validator(mode="after")
    def complete_partitions(self):
        defaults = {"train": "train", "validation": "validation", "test": "test"}
        self.partitions = {**defaults, **self.partitions}
        return self

    @model_validator(mode="after")
    def train_sets_held_out(self):
        if self.train_sets is None:
            return self
        keys = [(t.partition, t.fidelity) for t in self.train_sets]
        if len(set(keys)) != len(keys):
            raise ValueError("train_sets must be unique")
        held_out = {self.partitions["validation"], self.partitions["test"]}
        if any(t.partition in held_out for t in self.train_sets):
            raise ValueError("train_sets may not include the validation or test partition")
        return self

    @property
    def training_sets(self) -> list[tuple[str, str]]:
        """(partition, fidelity) of every training set."""
        if self.train_sets is None:
            return [(self.partitions["train"], "lf")]
        return [(t.partition, t.fidelity) for t in self.train_sets]

    @property
    def training_label(self) -> str:
        """Fidelity label of the training split: its common fidelity, otherwise "mixed"."""
        fidelities = {fidelity for _, fidelity in self.training_sets}
        return fidelities.pop() if len(fidelities) == 1 else "mixed"

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
                SurrogateConfig(
                    model=self.model, training=stage.training, selection=stage.selection
                )
                if stage.start_from != "initial":
                    parts = stage.start_from.split("/")
                    if len(parts) != 2 or parts[0] not in seen or parts[1] not in {"best", "final"}:
                        raise ValueError(
                            "Stage start_from must reference an earlier stage/best or final"
                        )
                if stage.training.backend == "bdt" and stage.start_from != "initial":
                    raise ValueError("BDT continuation is not supported")
                seen.add(stage.name)
        choices = {"best", "final", "pretraining"}
        choices.update(
            f"stages/{s.name}/{which}" for s in self.stages or [] for which in ("best", "final")
        )
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


# --- Neural field trained on ungrouped events (event store) -------------------------------


class StreamTraining(StrictConfigModel):
    batch_size: int = Field(default=65536, gt=0)
    n_passes: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    eval_every: int = Field(default=200, gt=0)  # optimizer steps between validations
    seed: int = 0
    device: Literal["cpu", "cuda"] = "cuda"
    precision: Literal["fp32", "bf16"] = "bf16"  # bf16 autocast on CUDA; CPU always fp32
    compile: bool = False  # torch.compile the event forward pass
    # Validation events used for checkpoint selection (the first ones of the validation
    # shards, which are already in random order); None = all.
    validation_events: int | None = Field(default=20_000_000, gt=0)


class StreamTreeTraining(StrictConfigModel):
    """Stream training of ``kind: bdt``: boosted trees fit an in-memory case-control sample.

    One pass over the training events keeps every positive and a seeded fraction of the
    negatives, about ``negatives_per_positive`` per positive; the kept negatives get weight
    1 / fraction, so the fitted probabilities stay calibrated. The boosting settings
    (``max_iter``, ``eval_every``, ...) are the model's ``architecture``.
    """

    negatives_per_positive: float = Field(default=20.0, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=65536, gt=0)  # records per read of the event source
    seed: int = 0
    # As in StreamTraining: validation events used for checkpoint selection; None = all.
    validation_events: int | None = Field(default=20_000_000, gt=0)


class DesignPointEvaluationSet(StrictConfigModel):
    """A prepared design point partition on which a stream-trained field is scored, under the split
    name used in metrics.json and the plots (``{fidelity}_{split}_{checkpoint}.npz``)."""

    split: Literal["validation", "test"]
    partition: str
    fidelity: Fidelity


class StreamRunConfig(StrictConfigModel):
    """An event model trained on every event of an event store, selected on its validation
    events by the Bernoulli log loss; evaluated later on design-point data (``predict``).

    ``dim_theta`` and ``dim_phi`` are the input widths of the design-point data it will be
    evaluated on, so the checkpoint loads like any field. A stream record holds the
    position (``dim_theta`` columns: theta + the position phi columns at evaluation), then
    the other phi columns in phi order, then the label.

    ``model: {kind: field}``: the neural field for point-cloud data (3-D direction for the
    spherical harmonics; see ``core.bernoulli_field``). ``kind: mlp`` / ``kind: transformer``:
    an MLP / tabular transformer on all normalized record columns, any number of them.
    ``kind: bdt``: boosted trees on an in-memory case-control sample of the records
    (``training`` is then a StreamTreeTraining). For all but the field, ``record_position``
    (old name ``mlp_position``) names the phi columns that are added to theta to give the
    record position (None: theta alone). No stream CNP: its context sets are the events of
    one design point, and ungrouped events have none (docs/stream.md).
    """

    model: FieldModel | MLPModel | TransformerModel | BDTModel = Field(discriminator="kind")
    training: StreamTraining | StreamTreeTraining = Field(default_factory=StreamTraining)
    event_store: Path
    # The application's event source ('module:Class', see core.event_stream) and its own
    # reading/shuffling options, passed to it unchanged.
    event_source: str
    event_source_options: dict[str, Any] = Field(default_factory=dict)
    # Design point data the field is evaluated on, with the metrics and plots of design point
    # training.
    data_provider: str | None = None  # None: NpzBatches
    data_directory: Path | None = None
    data_config: Path | None = None  # prepares data_directory when outdated
    evaluation: list[DesignPointEvaluationSet] = Field(default_factory=list)
    # Design points of the first validation set scored at every validation step (validation history
    # plot; seeded subset, None = all). Selection itself uses the validation events.
    history_design_points: int | None = Field(default=500, gt=0)
    evaluation_seed: int = 0
    # Label draws per design point for the within-design-point null on the evaluation sets.
    within_point_null_draws: int = Field(default=1000, ge=100)
    # Calibration per input column on the validation events and the evaluation sets.
    feature_calibration: FeatureCalibrationConfig = Field(default_factory=FeatureCalibrationConfig)
    # Residual learnability test on the validation events and the evaluation sets; None = off.
    residual_test: ResidualTestConfig | None = None
    unblind_test: bool = False
    output_directory: Path
    dim_theta: int = Field(ge=1)
    dim_phi: int = Field(ge=0)
    record_position: FieldPosition | None = Field(
        default=None, validation_alias=AliasChoices("record_position", "mlp_position")
    )

    @model_validator(mode="before")
    @classmethod
    def training_for_kind(cls, data):
        """Parse ``training`` as the settings of the model kind (trees or gradient steps)."""
        if not isinstance(data, dict):
            return data
        model = data.get("model")
        kind = model.get("kind") if isinstance(model, dict) else getattr(model, "kind", None)
        training = data.get("training")
        if training is None or isinstance(training, dict):
            schema = StreamTreeTraining if kind == "bdt" else StreamTraining
            try:
                data = {**data, "training": schema.model_validate(training or {})}
            except ValueError as exc:
                raise ValueError(f"training of a stream {kind}: {exc}") from exc
        return data

    @model_validator(mode="after")
    def training_matches_kind(self):
        trees = isinstance(self.model, BDTModel)
        if trees != isinstance(self.training, StreamTreeTraining):
            raise ValueError("kind: bdt takes StreamTreeTraining, the other kinds StreamTraining")
        return self

    @model_validator(mode="after")
    def evaluation_sets(self):
        if self.evaluation and self.data_directory is None:
            raise ValueError("evaluation needs data_directory")
        keys = [(e.split, e.fidelity) for e in self.evaluation]
        if len(set(keys)) != len(keys):
            raise ValueError("Each (split, fidelity) may be evaluated once")
        for item in self.evaluation:
            blinded = item.split == "test" or item.partition in TEST_PARTITIONS
            if blinded and not self.unblind_test:
                raise ValueError(
                    f"{item.partition} is blinded; set unblind_test: true once all choices "
                    "are final"
                )
        return self

    @property
    def network(self) -> FieldModel:
        """The network trained: the field, or for ``kind: mlp`` / ``transformer`` the
        unencoded field with that network. Not for ``kind: bdt`` (no network)."""
        if isinstance(self.model, FieldModel):
            return self.model
        if isinstance(self.model, BDTModel):
            raise ValueError("A stream BDT has no network; see position_columns")
        return FieldModel(
            architecture=self.model.architecture,
            position=self.record_position,
            encoding=RawPosition(),
            last_layer=self.model.last_layer,
        )

    @property
    def position_columns(self) -> list[int]:
        """phi columns added to theta to give the record position ([]: theta alone)."""
        position = (
            self.model.position if isinstance(self.model, FieldModel) else self.record_position
        )
        return [] if position is None else list(position.phi_columns)

    @model_validator(mode="after")
    def stream_columns(self):
        if self.record_position is not None and isinstance(self.model, FieldModel):
            raise ValueError(
                "record_position is for kind: mlp, transformer, bdt; a field uses model.position"
            )
        direction = getattr(self.model, "direction", None)
        used = self.position_columns + ([] if direction is None else direction.phi_columns)
        if used and max(used) >= self.dim_phi:
            raise ValueError(f"phi_columns {used} need dim_phi > {max(used)}")
        if self.position_columns and len(self.position_columns) != self.dim_theta:
            raise ValueError("position phi_columns must match dim_theta one to one")
        return self


def load_stream_config(path: str | Path) -> StreamRunConfig:
    """Stream field config; paths resolve relative to the YAML file."""
    path = Path(path).resolve()
    config = StreamRunConfig.model_validate(yaml.safe_load(path.read_text()))
    for name in ("event_store", "output_directory", "data_directory", "data_config"):
        value = getattr(config, name)
        if value is None:
            continue
        value = value.expanduser()
        setattr(config, name, value if value.is_absolute() else (path.parent / value).resolve())
    return config


# --- Residual learnability test on a finished run ------------------------------------------


class ResidualRunConfig(StrictConfigModel):
    """``python -m core.surrogates residual``: the residual learnability test on a finished
    training or stream run, from its saved logits and labels; the run is not modified.

    The inputs (θ, φ, stream records) are reloaded from the run's resolved config; their
    provenance and labels must match what the run recorded.
    """

    run: Path
    checkpoints: list[str] = Field(default_factory=lambda: ["best"], min_length=1)
    # Evaluated held-out sets as "split/fidelity", e.g. validation/hf or validation/stream
    # (stream validation events); None = every held-out set the run evaluated.
    sets: list[str] | None = Field(default=None, min_length=1)
    residual_test: ResidualTestConfig = Field(default_factory=ResidualTestConfig)
    output_directory: Path

    @field_validator("sets")
    @classmethod
    def set_names(cls, sets):
        for name in sets or []:
            split, _, fidelity = name.partition("/")
            if split not in ("validation", "test") or fidelity not in ("lf", "hf", "stream"):
                raise ValueError(f"sets entries are split/fidelity of held-out sets, not {name!r}")
        if sets is not None and len(set(sets)) != len(sets):
            raise ValueError("sets must be unique")
        return sets

    @field_validator("checkpoints")
    @classmethod
    def unique_checkpoints(cls, checkpoints):
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("checkpoints must be unique")
        return checkpoints


class WithinPointRunConfig(StrictConfigModel):
    """``python -m core.surrogates within_point``: the within-design-point metrics of the
    read-out (gain, skill, pair AUC, spread z, bootstrap intervals, null coverage) on a
    finished training or stream run, from its saved logits and labels, written into the
    run directory beside its read-out; the run's own files are not modified. For runs
    trained before these metrics existed.

    The target batches are reloaded from the run's resolved config; their provenance and
    labels must match what the run recorded. Design-point sets only (no stream events).
    """

    run: Path
    checkpoints: list[str] = Field(default_factory=lambda: ["best"], min_length=1)
    # Evaluated design-point sets as "split/fidelity", e.g. validation/lf; None = every
    # design-point set the run evaluated.
    sets: list[str] | None = Field(default=None, min_length=1)
    within_point_null_draws: int = Field(default=1000, ge=100)

    @field_validator("sets")
    @classmethod
    def set_names(cls, sets):
        for name in sets or []:
            split, _, fidelity = name.partition("/")
            if split not in ("validation", "test") or fidelity not in ("lf", "hf"):
                raise ValueError(
                    f"sets entries are split/fidelity of design-point sets, not {name!r}"
                )
        if sets is not None and len(set(sets)) != len(sets):
            raise ValueError("sets must be unique")
        return sets

    @field_validator("checkpoints")
    @classmethod
    def unique_checkpoints(cls, checkpoints):
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("checkpoints must be unique")
        return checkpoints


class ScoreRunConfig(StrictConfigModel):
    """``python -m core.surrogates score``: count-level predictive scores of a finished
    context-free event model with a Bayesian last layer on prepared design-point sets, the same
    measures as the binomial GP (predictive count NLL, plug-in NLL, MAE, RMSE, CRPS, Pearson r,
    coverage), with coverage and z-score plots. Predicts the sets anew; the run is not modified.

    Each design point is predicted from its own event inputs unless ``donors`` asks for the
    inputs of other design points (never its own), which needs the provider's
    ``transfer_events`` rule (``core/surrogates/datasets.py``).
    """

    run: Path
    checkpoints: list[str] = Field(default_factory=lambda: ["best"], min_length=1)
    # Provider partitions as "partition/fidelity", e.g. validation/hf (any prepared set; the
    # run's train sets are rejected).
    sets: list[str] = Field(min_length=1)
    # None: own event inputs. "scored": donors from each scored set itself. "partition/fidelity":
    # donors from that set. Donors need the provider's transfer_events.
    donors: str | None = None
    seed: int = 0  # donor draw
    threads: int = Field(default=1, ge=1)
    # Provider and data of the run's record unless given (e.g. another view of the same data).
    data_provider: str | None = "run"
    data_directory: Path | None = None
    # Coverage plot y-range; None: from the data.
    plot_ylim: tuple[float, float] | None = None
    # The run's own test partition is blinded unless set.
    unblind_test: bool = False
    output_directory: Path

    @field_validator("sets")
    @classmethod
    def set_names(cls, sets):
        for name in sets:
            partition, _, fidelity = name.partition("/")
            if not partition or not fidelity or "/" in fidelity:
                raise ValueError(f"sets entries are partition/fidelity, not {name!r}")
        if len(set(sets)) != len(sets):
            raise ValueError("sets must be unique")
        return sets

    @field_validator("donors")
    @classmethod
    def donor_name(cls, donors):
        if donors not in (None, "scored"):
            partition, _, fidelity = donors.partition("/")
            if not partition or not fidelity or "/" in fidelity:
                raise ValueError(f"donors is None, 'scored' or partition/fidelity, not {donors!r}")
        return donors

    @field_validator("checkpoints")
    @classmethod
    def unique_checkpoints(cls, checkpoints):
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("checkpoints must be unique")
        return checkpoints


def load_score_config(path: str | Path) -> ScoreRunConfig:
    """Score config; run, output and data paths resolve relative to the YAML file."""
    path = Path(path).resolve()
    config = ScoreRunConfig.model_validate(yaml.safe_load(path.read_text()))
    names = ("run", "output_directory") + (("data_directory",) if config.data_directory else ())
    return _resolve_run_paths(config, path, names=names)


def _resolve_run_paths(config, path, names=("run", "output_directory")):
    for name in names:
        value = getattr(config, name).expanduser()
        setattr(config, name, value if value.is_absolute() else (path.parent / value).resolve())
    return config


def load_residual_config(path: str | Path) -> ResidualRunConfig:
    """Residual-test config; run and output paths resolve relative to the YAML file."""
    path = Path(path).resolve()
    config = ResidualRunConfig.model_validate(yaml.safe_load(path.read_text()))
    return _resolve_run_paths(config, path)


def load_within_point_config(path: str | Path) -> WithinPointRunConfig:
    """Within-design-point config; the run path resolves relative to the YAML file."""
    path = Path(path).resolve()
    config = WithinPointRunConfig.model_validate(yaml.safe_load(path.read_text()))
    return _resolve_run_paths(config, path, names=("run",))
