"""Configuration for numerical feature-token attention."""
from typing import Literal
from pydantic import Field, model_validator
from schemas.config import StrictConfigModel


class TabularTransformerConfig(StrictConfigModel):
    architecture: Literal['legacy', 'ft_transformer'] = 'legacy'
    token_dim: int = Field(default=16, ge=2)
    n_heads: int = Field(default=2, ge=1)
    n_layers: int = Field(default=1, ge=1)
    feedforward_dim: int = Field(default=32, ge=1)
    dropout: float = Field(default=0., ge=0, lt=1)
    inference_chunk_size: int = Field(default=2048, ge=1)

    @model_validator(mode='after')
    def check_heads(self):
        if self.token_dim % self.n_heads:
            raise ValueError('token_dim must be divisible by n_heads')
        return self
