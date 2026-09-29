"""Single-logit tabular transformer: attention between features within each event."""
from pathlib import Path
import torch
from torch import nn
from schemas.tabular_transformer import TabularTransformerConfig


class FeatureTransformerBlock(nn.Module):
    """Pre-norm feature attention, ReGLU FFN; first attention has no norm.

    The last block needs only the CLS query. Other tokens remain keys/values.
    Design follows the FT-Transformer reference, with PyTorch attention.
    """
    def __init__(self, config, first):
        super().__init__()
        d = config.token_dim
        self.attention_norm = nn.Identity() if first else nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, config.n_heads,
                                               dropout=config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn_in = nn.Linear(d, 2*config.feedforward_dim)
        self.ffn_out = nn.Linear(config.feedforward_dim, d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens, cls_only=False):
        normalized = self.attention_norm(tokens)
        query = normalized[:, :1] if cls_only else normalized
        attended = self.attention(query, normalized, normalized, need_weights=False)[0]
        residual = tokens[:, :1] if cls_only else tokens
        tokens = residual + self.dropout(attended)
        a, b = self.ffn_in(self.ffn_norm(tokens)).chunk(2, dim=-1)
        return tokens + self.dropout(self.ffn_out(self.dropout(a*b.relu())))


class BernoulliTransformer(nn.Module):
    def __init__(self, config, dim_theta, dim_phi):
        super().__init__()
        self.config = config
        self.dim_theta, self.dim_phi = dim_theta, dim_phi
        features = (dim_theta or 0)+(dim_phi or 0)
        if features < 1:
            raise ValueError('At least one feature is required')
        self.feature_weight = nn.Parameter(torch.empty(features, config.token_dim))
        self.feature_bias = nn.Parameter(torch.empty(features, config.token_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.token_dim))
        nn.init.normal_(self.feature_weight, std=.02)
        nn.init.normal_(self.feature_bias, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        if config.architecture == 'ft_transformer':
            # Affine numeric embeddings initialized at a Linear-like scale.
            bound = config.token_dim**-.5
            nn.init.uniform_(self.feature_weight, -bound, bound)
            nn.init.uniform_(self.feature_bias, -bound, bound)
            nn.init.uniform_(self.cls_token, -bound, bound)
            self.blocks = nn.ModuleList([FeatureTransformerBlock(config, first=i == 0)
                                         for i in range(config.n_layers)])
            self.head = nn.Sequential(nn.LayerNorm(config.token_dim), nn.ReLU(),
                                      nn.Linear(config.token_dim, 1))
        else:
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(config.token_dim, config.n_heads,
                    config.feedforward_dim, config.dropout, activation='gelu',
                    batch_first=True, norm_first=True) for _ in range(config.n_layers)])
            self.head = nn.Sequential(nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1))

    def _logits(self, features):
        tokens = features[..., None]*self.feature_weight + self.feature_bias
        tokens = torch.cat([self.cls_token.expand(len(features), -1, -1), tokens], dim=1)
        for i, block in enumerate(self.blocks):
            if self.config.architecture == 'ft_transformer':
                tokens = block(tokens, cls_only=i == len(self.blocks)-1)
            else:
                tokens = block(tokens)
        return self.head(tokens[:, 0]).squeeze(-1)

    def forward(self, context, target):
        # Context deliberately unused: no cross-event or label conditioning.
        parts = []
        for name, dimension in [('theta', self.dim_theta), ('phi', self.dim_phi)]:
            values = getattr(target, name)
            if (values is None) != (dimension is None):
                raise ValueError(f'{name} presence does not match model')
            if values is not None:
                if values.shape[-1] != dimension:
                    raise ValueError(f'{name} dimension does not match model')
                tensor = torch.as_tensor(values, device=self.cls_token.device, dtype=self.cls_token.dtype)
                if name == 'theta':
                    tensor = tensor[:, None, :].expand(-1, target.n_events, -1)
                parts.append(tensor)
        features = torch.cat(parts, dim=-1).reshape(-1, (self.dim_theta or 0)+(self.dim_phi or 0))
        if self.training:
            logits = self._logits(features)
        else:
            logits = torch.cat([self._logits(chunk) for chunk in features.split(self.config.inference_chunk_size)])
        return logits.reshape(target.batch_size, target.n_events)

    def predict_beta(self, context, target):
        return self(context, target).sigmoid()


def save_transformer_checkpoint(path, model, *, encoder_config, dim_theta, dim_phi, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_kind='bernoulli_transformer_single_logit', model_state=model.state_dict(),
                    transformer_config=model.config.model_dump(), dim_theta=dim_theta,
                    dim_phi=dim_phi, metadata=metadata), path)


def load_transformer_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('model_kind') != 'bernoulli_transformer_single_logit':
        raise ValueError('Expected a tabular transformer checkpoint')
    model = BernoulliTransformer(TabularTransformerConfig(**payload['transformer_config']),
                                payload['dim_theta'], payload['dim_phi'])
    model.load_state_dict(payload['model_state'])
    return model, payload
