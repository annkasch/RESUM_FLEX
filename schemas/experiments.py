"""Versioned experiment and paired-comparison specifications."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from schemas.config import StrictConfigModel
from schemas.count_gp import CountGPBackendConfig
from schemas.projections import ProjectionConfig
from schemas.surrogates import (
    MFGPStageConfig,
    ModelSpec,
    SelectionMetric,
    SurrogateConfig,
    SurrogateRunConfig,
    TrainingSpec,
    TrainingStage,
)


class DatasetSpec(StrictConfigModel):
    prepared: Path | None = None
    preparation: Path | None = None
    format: Literal["events", "counts"] = "events"

    @model_validator(mode="after")
    def one_source(self):
        if (self.prepared is None) == (self.preparation is None):
            raise ValueError("Provide exactly one prepared dataset or preparation config")
        return self


class EvaluationSpec(StrictConfigModel):
    context_events: int = Field(default=64, ge=0)
    seed: int = 12345
    partitions: list[str] = Field(default_factory=lambda: ["validation/lf", "validation/hf"])
    weighting: Literal["equal_voxels"] = "equal_voxels"
    plots: list[
        Literal[
            "means",
            "residuals",
            "coverage",
            "precision_recall",
            "projected_axes",
            "projected_planes",
            "marginalized_axes",
            "marginalized_planes",
        ]
    ] = Field(
        default_factory=lambda: [
            "means",
            "residuals",
            "coverage",
            "precision_recall",
            "projected_axes",
            "projected_planes",
        ]
    )

    @model_validator(mode="after")
    def valid_partitions(self):
        choices = {f"{s}/{f}" for s in ("train", "validation", "test") for f in ("lf", "hf")}
        if not self.partitions or len(set(self.partitions)) != len(self.partitions):
            raise ValueError("Evaluation partitions must be nonempty and unique")
        if not set(self.partitions) <= choices:
            raise ValueError("Partition must be train|validation|test / lf|hf")
        return self


class EventPipeline(StrictConfigModel):
    kind: Literal["event"] = "event"
    model: ModelSpec
    training: TrainingSpec | None = None
    stages: list[TrainingStage] | None = Field(default=None, min_length=1)
    selection: SelectionMetric = "voxel_rate_mae"
    initial_checkpoint: Path | None = None
    fit: bool = True
    spatial_regression: MFGPStageConfig | None = None
    spatial_checkpoint: str = "best"

    @model_validator(mode="after")
    def valid_training(self):
        if self.training is None and self.stages:
            self.training = self.stages[0].training
        if self.fit and self.training is None:
            raise ValueError("Fitting an event model requires training or stages")
        if not self.fit and (self.initial_checkpoint is None or self.stages is not None):
            raise ValueError("Downstream-only runs need initial_checkpoint and no training stages")
        if self.training:
            SurrogateConfig(model=self.model, training=self.training, selection=self.selection)
            SurrogateRunConfig(
                model=self.model,
                training=self.training,
                stages=self.stages,
                data_directory=Path("."),
                output_directory=Path("."),
                mfgp_checkpoint=self.spatial_checkpoint,
            )
        return self


class CountPipeline(StrictConfigModel):
    kind: Literal["count_gp"] = "count_gp"
    backend: CountGPBackendConfig = Field(default_factory=CountGPBackendConfig)
    prediction_draws: int = Field(default=32768, ge=2048)
    prediction_seed: int = 42
    projections: ProjectionConfig = Field(default_factory=ProjectionConfig)


class ExperimentSpec(StrictConfigModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    hypothesis: str = ""
    tags: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    store: Path = Path("outputs/experiments")
    dataset: DatasetSpec
    pipeline: Annotated[EventPipeline | CountPipeline, Field(discriminator="kind")]
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)

    @model_validator(mode="after")
    def compatible(self):
        if self.pipeline.kind == "event":
            if self.dataset.format != "events" or self.evaluation.context_events == 0:
                raise ValueError("Event pipeline requires event arrays and a positive context size")
        elif self.dataset.format == "counts" and self.evaluation.context_events != 0:
            raise ValueError("Aggregated counts have no event IDs; use context_events=0")
        return self


class Variant(StrictConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    overrides: dict = Field(default_factory=dict)


class ComparisonSpec(StrictConfigModel):
    schema_version: Literal[1] = 1
    name: str
    baseline: Path
    variants: list[Variant] = Field(min_length=2)
    seeds: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique")
        if len({v.name for v in self.variants}) != len(self.variants):
            raise ValueError("Variant names must be unique")
        return self


def load_experiment(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if "pipeline" not in value:
        from core.experiments.migration import adapt_config

        return adapt_config(path)
    cfg = ExperimentSpec.model_validate(value)

    def resolve(p):
        return (
            p.expanduser().resolve()
            if p.expanduser().is_absolute()
            else (path.parent / p).resolve()
        )

    cfg.store = resolve(cfg.store)
    for field in ("prepared", "preparation"):
        if getattr(cfg.dataset, field) is not None:
            setattr(cfg.dataset, field, resolve(getattr(cfg.dataset, field)))
    if cfg.pipeline.kind == "event" and cfg.pipeline.initial_checkpoint:
        cfg.pipeline.initial_checkpoint = resolve(cfg.pipeline.initial_checkpoint)
    return cfg
