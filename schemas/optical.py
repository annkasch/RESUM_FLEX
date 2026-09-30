"""Validated configuration for fixed optical simulation datasets."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANOMALOUS_FILE = "lf/run067_x-0.088_y+0.484_z-1.462.stp.lh5"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(StrictModel):
    directory: Path | None = None
    directories: dict[Literal["lf", "hf"], list[Path]] = Field(default_factory=dict)
    exclude_files: list[str] = Field(default_factory=lambda: [ANOMALOUS_FILE])


    @model_validator(mode="after")
    def check_sources(self):
        if (self.directory is not None) == bool(self.directories):
            raise ValueError("Provide either directory or per-fidelity directories")
        if any(not paths for paths in self.directories.values()):
            raise ValueError("Each configured fidelity needs at least one folder")
        return self


class CylindricalConfig(StrictModel):
    include_azimuth: bool = True
    angle_encoding: Literal["sin_cos", "radians"] = "sin_cos"


class ThetaConfig(StrictModel):
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian"
    center_source: Literal["filename", "observed_midrange"] = "filename"
    cylindrical: CylindricalConfig = Field(default_factory=CylindricalConfig)


class VertexConfig(StrictModel):
    representation: Literal["offset", "absolute", "omit"] = "offset"
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian"


class MomentumConfig(StrictModel):
    representation: Literal["components", "direction_and_magnitude", "omit"] = "components"


class PhiConfig(StrictModel):
    vertex: VertexConfig = Field(default_factory=VertexConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)


class TargetConfig(StrictModel):
    kind: Literal["any_channel", "single_channel"] = "any_channel"
    detector_id: int | None = None

    @model_validator(mode="after")
    def check_detector(self):
        if (self.kind == "single_channel") != (self.detector_id is not None):
            raise ValueError("detector_id is required only for single_channel")
        return self


class SplitConfig(StrictModel):
    lf_train_only: bool = False
    hf_train_count: int | None = Field(default=None, gt=0)
    train_fraction: float = Field(default=0.7, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    test_fraction: float = Field(default=0.15, gt=0, lt=1)
    seed: int = 42
    manifest: Path = Path("splits/voxel_split.json")

    @model_validator(mode="after")
    def check_fractions(self):
        if abs(sum(self.fractions) - 1) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self

    @property
    def fractions(self) -> tuple[float, float, float]:
        return self.train_fraction, self.validation_fraction, self.test_fraction


class NormalizationConfig(StrictModel):
    method: Literal["standard", "minmax", "none"] = "standard"
    fit_on: Literal["training"] = "training"


class OpticalDataConfig(StrictModel):
    source: SourceConfig
    output_directory: Path = Path("outputs/optical_data")
    theta: ThetaConfig = Field(default_factory=ThetaConfig)
    phi: PhiConfig = Field(default_factory=PhiConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


def load_optical_config(path: str | Path) -> OpticalDataConfig:
    """Load preparation settings with paths relative to the configuration file."""
    import yaml

    path = Path(path).resolve()
    config = OpticalDataConfig.model_validate(yaml.safe_load(path.read_text()))

    def resolve(value):
        value = value.expanduser()
        return value.resolve() if value.is_absolute() else (path.parent / value).resolve()

    if config.source.directory is not None:
        config.source.directory = resolve(config.source.directory)
    config.source.directories = {
        fidelity: [resolve(folder) for folder in folders]
        for fidelity, folders in config.source.directories.items()
    }
    config.output_directory = resolve(config.output_directory)
    config.split.manifest = resolve(config.split.manifest)
    return config
