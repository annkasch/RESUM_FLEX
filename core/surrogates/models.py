"""Adapters preserve the tested architectures and keep tree fitting separate."""

import numpy as np
from pydantic import TypeAdapter

from core.surrogates.base import EventPrediction, EventSurrogate
from schemas.surrogates import ModelSpec


def slice_batch(batch, start, stop):
    if batch is None:
        return None
    return batch.model_copy(
        update={
            "theta": None if batch.theta is None else batch.theta[start:stop],
            "phi": None if batch.phi is None else batch.phi[start:stop],
            "labels": batch.labels[start:stop],
        }
    )


class NeuralSurrogate(EventSurrogate):
    def __init__(self, config, dim_theta, dim_phi):
        super().__init__(config, dim_theta, dim_phi)
        self.uses_context = config.kind == "cnp"
        if config.kind == "cnp":
            from core.bernoulli_cnp import BernoulliCNP

            self.module = BernoulliCNP(config.encoder, dim_theta, dim_phi)
        elif config.kind == "mlp":
            from core.bernoulli_mlp import BernoulliMLP
            from schemas.config import EncoderConfig

            encoder = EncoderConfig(type="mlp", latent_dim=1, **config.architecture.model_dump())
            self.module = BernoulliMLP(encoder, dim_theta, dim_phi)
        else:
            from core.bernoulli_transformer import BernoulliTransformer

            self.module = BernoulliTransformer(config.architecture, dim_theta, dim_phi)

    def predict(self, target, *, context=None):
        import torch

        self.validate(target, context)
        training = self.module.training
        try:
            self.module.eval()
            with torch.no_grad():
                chunks = [
                    self.module(slice_batch(context, i, i + 4), slice_batch(target, i, i + 4))
                    .detach()
                    .cpu()
                    .numpy()
                    for i in range(0, target.batch_size, 4)
                ]
            return EventPrediction(np.concatenate(chunks))
        finally:
            self.module.train(training)


class TreeSurrogate(EventSurrogate):
    def __init__(self, config, dim_theta, dim_phi):
        super().__init__(config, dim_theta, dim_phi)
        try:
            from core.surrogate_bdt import build_bdt

            self.estimator = build_bdt(config.architecture)
        except ImportError as exc:
            raise ImportError(
                'BDT requires the optional dependency: pip install -e ".[bdt]"'
            ) from exc

    def predict(self, target, *, context=None):
        from core.surrogate_bdt import event_features

        self.validate(target, context)
        logits = self.estimator.decision_function(event_features(target))
        return EventPrediction(np.asarray(logits).reshape(target.batch_size, target.n_events))


def build_surrogate(config, dim_theta, dim_phi, *, seed=0):
    """Construct any supported model from a typed spec or a kind-tagged dict."""
    config = TypeAdapter(ModelSpec).validate_python(config)
    if config.kind == "bdt":
        return TreeSurrogate(config, dim_theta, dim_phi)
    import torch

    # Reproducible initialization without changing the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return NeuralSurrogate(config, dim_theta, dim_phi)
