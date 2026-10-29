"""Validated configuration for fixed optical simulation datasets."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANOMALOUS_FILE = "lf/run067_x-0.088_y+0.484_z-1.462.stp.lh5"
# Roles of voxels in the role-based layout; see RoleSource. New roles are appended.
ROLES = (
    "cnp_train", "cnp_validation", "cnp_test", "gp_train", "gp_unlabeled", "gp_validation",
    "gp_test",
)
Role = Literal[
    "cnp_train", "cnp_validation", "cnp_test", "gp_train", "gp_unlabeled", "gp_validation",
    "gp_test",
]
# Roles whose files need no optical hits (inputs only, e.g. for event-model inference).
UNLABELED_ROLES = ("gp_unlabeled",)
# Roles that may share voxel positions across folders (nested multi-fidelity GP design,
# e.g. HF training voxels at positions of unlabeled LF voxels). Any other shared position
# between different roles is rejected as leakage.
GP_TRAINING_ROLES = ("gp_train", "gp_unlabeled")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleSource(StrictModel):
    """One folder whose voxels are divided among ROLES by fraction or by file count.

    Roles left out get no files. ``fractions`` must sum to 1 and are rounded per folder;
    ``counts`` must add up to the folder's file count unless one role takes ``rest``.
    An already split folder is e.g. ``fractions: {cnp_validation: 1.0}``.
    """

    path: Path
    fidelity: Literal["lf", "hf"] = "lf"
    fractions: dict[Role, float] | None = None
    counts: dict[Role, int | Literal["rest"]] | None = None

    @model_validator(mode="after")
    def check_split(self):
        if (self.fractions is None) == (self.counts is None):
            raise ValueError(f"{self.path}: give either fractions or counts")
        if self.fractions is not None:
            values = list(self.fractions.values())
            if any(not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-9:
                raise ValueError(f"{self.path}: fractions must lie in [0, 1] and sum to 1")
        else:
            values = list(self.counts.values())
            if values.count("rest") > 1:
                raise ValueError(f"{self.path}: only one role may take 'rest'")
            if any(v != "rest" and v < 0 for v in values):
                raise ValueError(f"{self.path}: counts must be nonnegative")
        return self


class SourceConfig(StrictModel):
    directory: Path | None = None
    directories: dict[Literal["lf", "hf"], list[Path]] = Field(default_factory=dict)
    # Role-based LF layout: per-folder splits replace the global split fractions.
    sources: list[RoleSource] = Field(default_factory=list)
    exclude_files: list[str] = Field(default_factory=lambda: [ANOMALOUS_FILE])

    @model_validator(mode="after")
    def check_sources(self):
        if sum([self.directory is not None, bool(self.directories), bool(self.sources)]) != 1:
            raise ValueError("Provide either directory, per-fidelity directories or role sources")
        if any(not paths for paths in self.directories.values()):
            raise ValueError("Each configured fidelity needs at least one folder")
        paths = [s.path for s in self.sources]
        if len(set(paths)) != len(paths):
            raise ValueError("Each role source folder may appear only once")
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
    validation_fraction: float = Field(default=0.15, ge=0, lt=1)
    test_fraction: float = Field(default=0.15, ge=0, lt=1)
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
    fit_fidelities: list[Literal["lf", "hf"]] | None = Field(default=None, min_length=1)


class OpticalDataConfig(StrictModel):
    source: SourceConfig
    output_directory: Path = Path("outputs/optical_data")
    # Shared float32 copy of the raw files (data/event_cache.py); by default next to
    # output_directory, so data configs writing to the same parent folder share it.
    cache_directory: Path | None = None
    theta: ThetaConfig = Field(default_factory=ThetaConfig)
    phi: PhiConfig = Field(default_factory=PhiConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)

    @model_validator(mode="after")
    def check_role_split(self):
        per_source = {"seed", "manifest"}
        default = SplitConfig().model_dump(exclude=per_source)
        if self.source.sources and self.split.model_dump(exclude=per_source) != default:
            raise ValueError(
                "With role sources, set splits per source; only split.seed and split.manifest apply"
            )
        return self

    @property
    def event_cache(self) -> Path:
        """Directory of the shared event cache."""
        return self.cache_directory or self.output_directory.parent / "event_cache"


def resolve_cache_directory(config: OpticalDataConfig, resolve) -> None:
    if config.cache_directory is not None:
        config.cache_directory = resolve(config.cache_directory)


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
    for source in config.source.sources:
        source.path = resolve(source.path)
    config.output_directory = resolve(config.output_directory)
    resolve_cache_directory(config, resolve)
    config.split.manifest = resolve(config.split.manifest)
    return config
