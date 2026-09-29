"""Configuration for the event-level boosted decision-tree baseline."""
from pydantic import Field
from schemas.config import StrictConfigModel


class BDTConfig(StrictConfigModel):
    learning_rate: float = Field(default=.05, gt=0)
    max_iter: int = Field(default=1000, gt=0)
    eval_every: int = Field(default=25, gt=0)
    max_leaf_nodes: int = Field(default=15, ge=2)
    max_depth: int = Field(default=4, ge=1)
    min_samples_leaf: int = Field(default=100, ge=1)
    l2_regularization: float = Field(default=1., ge=0)
    max_bins: int = Field(default=255, ge=2, le=255)
    seed: int = 0
