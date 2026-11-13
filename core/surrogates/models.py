"""Adapters preserve the tested architectures and keep tree fitting separate."""

import numpy as np
from pydantic import TypeAdapter

from core.surrogates.base import EventPrediction, EventSurrogate
from schemas.surrogates import ModelSpec, uses_context


def slice_batch(batch, start, stop):
    if batch is None:
        return None
    return batch.model_copy(
        update={
            "theta": None if batch.theta is None else batch.theta[start:stop],
            "phi": None if batch.phi is None else batch.phi[start:stop],
            "labels": batch.labels[start:stop],
            "event_counts": None if batch.event_counts is None else batch.event_counts[start:stop],
        }
    )


class NeuralSurrogate(EventSurrogate):
    def __init__(self, config, dim_theta, dim_phi):
        super().__init__(config, dim_theta, dim_phi)
        self.uses_context = uses_context(config.kind)
        if config.kind == "cnp":
            from core.bernoulli_cnp import BernoulliCNP

            self.module = BernoulliCNP(config.encoder, dim_theta, dim_phi)
        elif config.kind == "legacy_cnp":
            from core.surrogate_cnp import build_cnp

            self.module = build_cnp(config.encoder, dim_theta, dim_phi)
        elif config.kind == "mlp":
            from core.bernoulli_mlp import BernoulliMLP
            from schemas.config import EncoderConfig

            encoder = EncoderConfig(type="mlp", latent_dim=1, **config.architecture.model_dump())
            self.module = BernoulliMLP(encoder, dim_theta, dim_phi)
        elif config.kind == "field":
            from core.bernoulli_field import BernoulliField

            self.module = BernoulliField(config, dim_theta, dim_phi)
        else:
            from core.bernoulli_transformer import BernoulliTransformer

            self.module = BernoulliTransformer(config.architecture, dim_theta, dim_phi)

    # Fitted Laplace posterior of the final layer (core.last_layer), or None.
    last_layer = None

    @property
    def supports_uncertainty(self):
        return self.last_layer is not None

    @property
    def capabilities(self):
        return {**super().capabilities, "predictive_distribution": self.supports_uncertainty}

    def configure_normalization(self, normalization):
        """Give a field the data normalization it needs to combine inputs in physical units.

        No-op for other kinds. Stored in the module state, so checkpoints keep it.
        """
        if self.config.kind == "field":
            self.module.set_normalization(normalization)

    def output_layer(self):
        """Explicit final projection for each architecture; never infer by ordering."""
        if self.config.kind == "transformer":
            return self.module.head[-1]
        decoder = self.module.decoder
        if self.config.kind == "field" and hasattr(decoder, "transformer"):
            return decoder.head[-1]  # field with a transformer network
        return decoder.net[-1]

    def predict(self, target, *, context=None):
        import torch

        self.validate(target, context)
        training = self.module.training
        try:
            self.module.eval()
            with torch.no_grad():
                chunks, scales = [], []
                for i in range(0, target.batch_size, 4):
                    out = self.module(slice_batch(context, i, i + 4), slice_batch(target, i, i + 4))
                    if self.config.kind == "legacy_cnp":
                        from core.surrogate_cnp import resum_binary_logits, resum_binary_moments

                        _, scale = resum_binary_moments(out)
                        scales.append(scale.detach().cpu().numpy())
                        out = resum_binary_logits(out)
                    chunks.append(out.detach().cpu().numpy())
            return EventPrediction(
                np.concatenate(chunks),
                np.concatenate(scales) if scales else None,
                event_counts=target.event_counts,
            )
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

    # Stream-trained trees: phi columns added to theta to give the record position ([]: theta
    # alone); None for trees trained on design points. Saved with the checkpoint.
    stream_position = None
    _data_normalization = None

    @property
    def uses_stream_units(self) -> bool:
        return self.stream_position is not None

    @property
    def n_features(self) -> int:
        """Input columns of the estimator: theta + phi, or the stream record width."""
        width = (self.dim_theta or 0) + (self.dim_phi or 0)
        return width - len(self.stream_position or [])

    def configure_normalization(self, normalization):
        """Stream-trained trees map design-point inputs back to physical record columns."""
        if self.uses_stream_units:
            if normalization is None:
                raise ValueError("A stream-trained BDT needs the design-point data normalization")
            self._data_normalization = normalization

    def features(self, target):
        from core.surrogate_bdt import event_features, stream_records

        if not self.uses_stream_units:
            return event_features(target)
        if self._data_normalization is None:
            raise ValueError("Call configure_normalization before scoring design points")
        return stream_records(target, self._data_normalization, self.stream_position)

    def predict(self, target, *, context=None):
        self.validate(target, context)
        logits = self.estimator.decision_function(self.features(target))
        return EventPrediction(
            np.asarray(logits).reshape(target.batch_size, target.n_events),
            event_counts=target.event_counts,
        )


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
