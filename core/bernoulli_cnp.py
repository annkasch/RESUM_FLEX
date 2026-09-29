"""Mean-only CNP with one Bernoulli logit per target event."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core.networks import MLPEncoder, build_encoder
from core.surrogate_cnp import ConditionalNeuralProcess, ContextPointEncoder
from schemas.config import EncoderConfig


class BernoulliCNP(ConditionalNeuralProcess):
    """Reuse CNP encoding/aggregation, with a genuinely single-output decoder."""

    def __init__(self, encoder_config, dim_theta, dim_phi):
        z = encoder_config.latent_dim
        super().__init__(
            build_encoder(encoder_config, dim_theta, dim_phi),
            ContextPointEncoder(z, encoder_config.hidden_dims, z, encoder_config.dropout),
            MLPEncoder(2*z, encoder_config.hidden_dims, 1, encoder_config.dropout),
        )

    def forward(self, context, target):
        if context.batch_size != target.batch_size or context.mode is not target.mode:
            raise ValueError('Context and target must have matching voxel count and mode')
        representation = self.aggregate(context)
        _, features = self._encode_per_event(target)
        expanded = representation[:, None, :].expand(-1, target.n_events, -1)
        return self.decoder(torch.cat([expanded, features], dim=-1)).squeeze(-1)

    def predict_beta(self, context, target):
        return self(context, target).sigmoid()


def evaluate_bernoulli(model, context, target):
    """Exact fixed-target mean diagnostics; no Gaussian uncertainty approximation."""
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            logits = model(context, target)
            labels = torch.as_tensor(target.labels, dtype=logits.dtype, device=logits.device)
            p = logits.sigmoid()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite validation logits')
            predicted = p.mean(1).cpu().numpy().astype(float)
            observed = target.labels.mean(1)
            residual = predicted-observed
            correlation = (float(np.corrcoef(predicted,observed)[0,1])
                           if predicted.std()>0 and observed.std()>0 else None)
            summary = dict(voxels=len(observed), target_events_per_voxel=target.n_events,
                mean_observed=float(observed.mean()), mean_predicted=float(predicted.mean()),
                ratio_of_means=float(predicted.mean()/observed.mean()) if observed.mean()>0 else None,
                pearson_r=correlation, voxel_rate_mae=float(np.abs(residual).mean()),
                voxel_rate_rmse=float(np.sqrt(np.square(residual).mean())),
                bernoulli_log_loss=float(F.binary_cross_entropy_with_logits(logits.double(),labels.double())),
                brier_score=float(((p.double()-labels.double())**2).mean()),
                min_predicted_rate=float(predicted.min()), max_predicted_rate=float(predicted.max()))
    finally:
        model.train(was_training)
    return summary, dict(observed=observed,predicted=predicted)


def save_bernoulli_checkpoint(path, model, *, encoder_config, dim_theta, dim_phi, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_kind='bernoulli_cnp_single_logit', model_state=model.state_dict(),
        encoder_config=encoder_config.model_dump(), dim_theta=dim_theta, dim_phi=dim_phi,
        metadata=metadata),path)


def load_bernoulli_checkpoint(path):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('model_kind') != 'bernoulli_cnp_single_logit':
        raise ValueError('Expected a single-logit Bernoulli CNP checkpoint')
    model = BernoulliCNP(EncoderConfig(**payload['encoder_config']),payload['dim_theta'],payload['dim_phi'])
    model.load_state_dict(payload['model_state'])
    return model,payload
