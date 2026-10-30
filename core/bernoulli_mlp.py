"""Event MLP with one raw Bernoulli logit; no context conditioning."""
import torch
from torch import nn
from core.networks import MLPEncoder


class BernoulliMLP(nn.Module):
    """Concatenate normalized voxel theta and event phi, then predict a logit.

    The context argument only provides compatibility with the matched experiment
    runner. Neither context features nor any labels enter predictions.
    """
    def __init__(self, encoder_config, dim_theta, dim_phi):
        super().__init__()
        self.dim_theta, self.dim_phi = dim_theta, dim_phi
        self.decoder = MLPEncoder((dim_theta or 0)+(dim_phi or 0),
                                  encoder_config.hidden_dims, 1, encoder_config.dropout)

    def forward(self, context, target):
        parameter = next(self.parameters())
        parts = []
        for name, dimension in [('theta', self.dim_theta), ('phi', self.dim_phi)]:
            values = getattr(target, name)
            if (values is None) != (dimension is None):
                raise ValueError(f'{name} presence must match model input dimensions')
            if values is not None:
                if values.shape[-1] != dimension:
                    raise ValueError(f'{name} dimension does not match model')
                tensor = torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device)
                if name == 'theta':
                    tensor = tensor[:, None, :].expand(-1, target.n_events, -1)
                parts.append(tensor)
        return self.decoder(torch.cat(parts, dim=-1)).squeeze(-1)

    def predict_beta(self, context, target):
        return self(context, target).sigmoid()
