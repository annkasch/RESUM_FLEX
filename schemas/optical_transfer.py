"""Optical training and independent testing at different simulation budgets."""

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from schemas.optical import OpticalDataConfig, SourceConfig


class OpticalTransferDataConfig(OpticalDataConfig):
    test_source: SourceConfig
    training_primaries: int = Field(default=1500, gt=1)
    hf_primaries: int = Field(default=5000, gt=1)
    test_primaries: int = Field(default=1000, gt=1)
    spatial_separation_mm: float = Field(default=5.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def transfer_split(self):
        if self.split.test_fraction != 0 or self.split.validation_fraction <= 0:
            raise ValueError("Use a train/validation split; test files come from test_source")
        if self.split.lf_train_only:
            raise ValueError("Cross-budget testing requires LF training and validation")
        if self.source.directory is not None or "lf" not in self.source.directories:
            raise ValueError("Provide source directory lists including LF")
        if self.test_source.directory is not None or set(self.test_source.directories) != {"lf"}:
            raise ValueError("Provide LF-only directory lists for test_source")
        if self.split.hf_train_count is not None and "hf" not in self.source.directories:
            raise ValueError("hf_train_count requires HF source directories")
        return self


def load_optical_transfer_config(path):
    path = Path(path).resolve()
    config = OpticalTransferDataConfig.model_validate(yaml.safe_load(path.read_text()))

    def resolve(value):
        value = value.expanduser()
        return value.resolve() if value.is_absolute() else (path.parent / value).resolve()

    for source in (config.source, config.test_source):
        source.directories = {
            fid: [resolve(folder) for folder in folders]
            for fid, folders in source.directories.items()
        }
    config.output_directory = resolve(config.output_directory)
    config.split.manifest = resolve(config.split.manifest)
    return config
