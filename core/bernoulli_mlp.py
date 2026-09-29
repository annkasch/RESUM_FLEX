"""Event MLP with one raw Bernoulli logit; no context conditioning."""
from pathlib import Path
import torch
from torch import nn
from core.networks import MLPEncoder
from schemas.config import EncoderConfig


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


def save_mlp_checkpoint(path, model, *, encoder_config, dim_theta, dim_phi, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_kind='bernoulli_mlp_single_logit', model_state=model.state_dict(),
                    encoder_config=encoder_config.model_dump(), dim_theta=dim_theta,
                    dim_phi=dim_phi, metadata=metadata), path)


def load_mlp_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('model_kind') != 'bernoulli_mlp_single_logit':
        raise ValueError('Expected a single-logit Bernoulli MLP checkpoint')
    model = BernoulliMLP(EncoderConfig(**payload['encoder_config']),
                         payload['dim_theta'], payload['dim_phi'])
    model.load_state_dict(payload['model_state'])
    return model, payload
