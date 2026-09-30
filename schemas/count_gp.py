"""Configuration for a shared-probability GP trained directly on optical counts."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from schemas.optical import OpticalDataConfig, StrictModel


class CountGPBackendConfig(StrictModel):
    kind: Literal["binomial_laplace"] = "binomial_laplace"
    kernel: Literal["rbf", "matern52"] = "matern52"
    max_iters: int = Field(default=300, gt=0)
    n_restarts: int = Field(default=2, gt=0)
    seed: int = 42


class CountGPConfig(StrictModel):
    data: OpticalDataConfig
    backend: CountGPBackendConfig = Field(default_factory=CountGPBackendConfig)
    output_directory: Path = Path("outputs/optical_binomial")
    prediction_draws: int = Field(default=32768, ge=2048)
    prediction_seed: int = 42

    @model_validator(mode="after")
    def validate_coordinates(self):
        if self.data.theta.coordinates != "cartesian":
            raise ValueError("Count GP requires Cartesian voxel coordinates")
        if self.data.normalization.method != "standard":
            raise ValueError("Count GP uses training-only standard coordinate scaling")
        return self


def load_count_gp_config(path):
    path = Path(path).resolve()
    cfg = CountGPConfig.model_validate(yaml.safe_load(path.read_text()))

    def resolve(value):
        value = value.expanduser()
        return value.resolve() if value.is_absolute() else (path.parent / value).resolve()

    cfg.output_directory = resolve(cfg.output_directory)
    cfg.data.output_directory = resolve(cfg.data.output_directory)
    cfg.data.split.manifest = resolve(cfg.data.split.manifest)
    if cfg.data.source.directory is not None:
        cfg.data.source.directory = resolve(cfg.data.source.directory)
    cfg.data.source.directories = {
        role: [resolve(p) for p in folders] for role, folders in cfg.data.source.directories.items()
    }
    return cfg
