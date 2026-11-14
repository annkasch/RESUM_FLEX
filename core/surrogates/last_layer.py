"""Bayesian last layer for neural single-logit event models (mlp, cnp, field, transformer).

After training, the final linear layer gets a Laplace posterior (core.last_layer) from one
pass over the natural training events; nothing is retrained. Predictions then carry
uncertainty: weight draws give, per design point, draws of the mean hit probability over that
design point's events. The draws share the weights across all events and design points, so the
correlation of the uncertainty is kept (it does not average down over a design point's events).
"""

import numpy as np
from scipy.special import expit

from core.last_layer import LastLayerPosterior, laplace_posterior
from core.surrogates.models import slice_batch

SUPPORTED_KINDS = ("mlp", "cnp", "field", "transformer")


def captured_features(captured, shape):
    """Final-layer inputs of one forward pass, joined over its chunks: [*shape, H].

    The transformer calls its head once per ``inference_chunk_size`` events, so a pass can
    capture several [m, H] blocks (in event order); the other networks capture one block.
    """
    import torch

    features = (
        captured[0]
        if len(captured) == 1
        else torch.cat([c.reshape(-1, c.shape[-1]) for c in captured])
    )
    return features.reshape(*shape, features.shape[-1])


def _layer(model):
    import torch

    layer = model.output_layer()
    if not isinstance(layer, torch.nn.Linear) or layer.out_features != 1:
        raise ValueError("A Bayesian last layer needs a final Linear layer with one output")
    return layer


def last_layer_weights(model) -> np.ndarray:
    """Trained weights [H + 1] = [weights..., bias] of the final layer."""
    layer = _layer(model)
    weight = layer.weight.detach().double().cpu().numpy().ravel()
    bias = 0.0 if layer.bias is None else float(layer.bias.detach().double().cpu())
    return np.append(weight, bias)


def feature_chunks(model, target, context=None, *, chunk=4):
    """Yield (start, stop, features [b, N, H + 1], valid [b, N]) of the final layer's input.

    The bias column is appended. Dropout is off (eval mode); gradients are not tracked.
    """
    import torch

    layer = _layer(model)
    captured = []
    handle = layer.register_forward_hook(lambda _m, inputs, _o: captured.append(inputs[0]))
    training = model.module.training
    try:
        model.module.eval()
        with torch.no_grad():
            for start in range(0, target.batch_size, chunk):
                stop = min(start + chunk, target.batch_size)
                captured.clear()
                model.module(slice_batch(context, start, stop), slice_batch(target, start, stop))
                valid = target.event_mask[start:stop]  # [b, N]
                features = captured_features(captured, valid.shape).double().cpu().numpy()
                ones = np.ones((*features.shape[:-1], 1))  # features: [b, N, H]
                yield start, stop, np.concatenate([features, ones], axis=-1), valid
    finally:
        handle.remove()
        model.module.train(training)


def fit_last_layer(model, train, context=None) -> LastLayerPosterior:
    """Laplace posterior from all valid events of ``train`` (natural labels, weight 1).

    ``context`` is required for context models (the CNP); its events condition the
    features of ``train``'s events and are not counted in the Hessian.
    """
    settings = model.config.last_layer
    if model.config.kind not in SUPPORTED_KINDS or settings is None:
        raise ValueError(f"{model.config.kind} has no Bayesian last layer configured")
    if model.uses_context and context is None:
        raise ValueError("A context model needs context events for the last-layer fit")
    model.validate(train, context)
    if not np.isin(train.labels, [0, 1]).all():
        raise ValueError("The last-layer fit needs real binary labels")
    mean = last_layer_weights(model)
    hessian = np.zeros((len(mean), len(mean)))
    events = 0
    for _start, _stop, features, valid in feature_chunks(model, train, context):
        f = features[valid]  # [E, H + 1]
        p = expit(f @ mean)
        hessian += (f * (p * (1 - p))[:, None]).T @ f
        events += len(f)
    posterior = laplace_posterior(
        hessian, mean, prior_precision=settings.prior_precision, events=events
    )
    model.last_layer = posterior
    model.metadata["last_layer"] = dict(
        prior_precision=posterior.prior_precision,
        prior_precision_source=("evidence" if settings.prior_precision == "evidence" else "fixed"),
        log_evidence=posterior.log_evidence,
        events=events,
        parameters=len(mean),
        hessian="exact Bernoulli Hessian of the final layer over natural training events",
    )
    return posterior


def design_point_probability_draws(model, target, context=None, *, n_draws=None, seed=None):
    """Draws [S, B] of each design point's mean hit probability over its valid events.

    Defaults to the configured ``last_layer.draws`` and ``last_layer.seed``.
    """
    posterior = getattr(model, "last_layer", None)
    if posterior is None:
        raise ValueError("The model has no fitted last-layer posterior")
    settings = model.config.last_layer
    weights = posterior.sample(
        settings.draws if n_draws is None else n_draws,
        settings.seed if seed is None else seed,
    )  # [H + 1, S]
    model.validate(target, context)
    out = np.empty((weights.shape[1], target.batch_size))
    for start, stop, features, valid in feature_chunks(model, target, context):
        logits = features @ weights  # [b, N, S]
        p = expit(logits)
        p = np.where(valid[..., None], p, 0.0).sum(axis=1)  # [b, S]
        out[:, start:stop] = (p / valid.sum(axis=1)[:, None]).T
    return out
