"""Boosted binary classifier over design point coordinates and event features.

Trained on design points it sees normalized [theta, phi]; trained on an event stream
(``stream_records``) it sees the stream record columns in physical units.
"""

import numpy as np
from scipy.special import expit

from core.precision_recall import precision_recall


def event_features(batch):
    """Flatten [design_point,event] order without using labels or context statistics."""
    parts = []
    if batch.theta is not None:
        parts.append(
            np.broadcast_to(
                batch.theta[:, None, :], (batch.batch_size, batch.n_events, batch.theta.shape[-1])
            )
        )
    if batch.phi is not None:
        parts.append(batch.phi)
    if not parts:
        raise ValueError("BDT requires at least one feature")
    return np.concatenate(parts, axis=-1).reshape(batch.batch_size * batch.n_events, -1)


def _affine(normalization, group, size):
    """(offset, scale) [size] of an affine normalization x_n = (x - offset) / scale."""
    entry = normalization.get(group) if isinstance(normalization, dict) else None
    if not isinstance(entry, dict) or not {"offset", "scale"} <= entry.keys():
        raise ValueError(f"A stream BDT needs an affine {group} normalization (offset, scale)")
    offset = np.asarray(entry["offset"], dtype=np.float64)
    scale = np.asarray(entry["scale"], dtype=np.float64)
    if offset.shape != (size,) or scale.shape != (size,):
        raise ValueError(f"{group} normalization does not match {size} input columns")
    return offset, scale


def stream_records(batch, normalization, position_columns):
    """Design-point inputs as stream record columns [B * N, W] in physical units.

    Position = theta + phi[position_columns] (theta alone for []), then the other phi columns
    in phi order: the layout a stream BDT was trained on (docs/stream.md). ``normalization``
    is the data's affine normalization of theta and phi.
    """
    if batch.theta is None:
        raise ValueError("A stream BDT needs theta (the record position)")
    b, n = batch.batch_size, batch.n_events
    theta_offset, theta_scale = _affine(normalization, "theta", batch.theta.shape[-1])
    position = np.broadcast_to(
        (batch.theta * theta_scale + theta_offset)[:, None, :], (b, n, batch.theta.shape[-1])
    )  # [B, N, T]
    parts = [position]
    if batch.phi is not None:
        dim_phi = batch.phi.shape[-1]
        phi_offset, phi_scale = _affine(normalization, "phi", dim_phi)
        phi = batch.phi * phi_scale + phi_offset  # [B, N, P]
        if position_columns:
            parts[0] = position + phi[..., position_columns]
        parts.append(phi[..., [c for c in range(dim_phi) if c not in position_columns]])
    elif position_columns:
        raise ValueError("position_columns need phi")
    return np.concatenate(parts, axis=-1).reshape(b * n, -1)


def build_bdt(config):
    # Optional dependency: importing the other surrogates does not require sklearn.
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.learning_rate,
        max_iter=config.eval_every,
        max_leaf_nodes=config.max_leaf_nodes,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,
        categorical_features=None,
        early_stopping=False,
        warm_start=True,
        random_state=config.seed,
        class_weight=None,
    )


def score_bdt(model, batch):
    batch.require_rectangular("score_bdt (use core.surrogates.evaluate_surrogate)")
    scores = np.asarray(model.decision_function(event_features(batch)), dtype=float)
    labels = batch.labels.ravel()
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite BDT scores")
    probabilities = expit(scores)
    observed = batch.labels.mean(1)
    predicted = probabilities.reshape(batch.batch_size, batch.n_events).mean(1)
    pr = precision_recall(labels, scores)
    residual = predicted - observed
    metrics = dict(
        design_points=batch.batch_size,
        events=len(labels),
        positives=int(labels.sum()),
        mean_observed=float(observed.mean()),
        mean_predicted=float(predicted.mean()),
        ratio_of_means=float(predicted.mean() / observed.mean()),
        pearson_r=float(np.corrcoef(observed, predicted)[0, 1])
        if predicted.std() > 0 and observed.std() > 0
        else None,
        design_point_rate_mae=float(np.abs(residual).mean()),
        design_point_rate_rmse=float(np.sqrt(np.square(residual).mean())),
        bernoulli_log_loss=float(
            np.where(labels == 1, np.logaddexp(0, -scores), np.logaddexp(0, scores)).mean()
        ),
        brier_score=float(np.square(probabilities - labels).mean()),
        average_precision=pr["average_precision"],
        prevalence=pr["prevalence"],
    )
    arrays = dict(
        logits=scores,
        labels=labels,
        observed=observed,
        predicted=predicted,
        precision=pr["precision"],
        recall=pr["recall"],
        logit_threshold=pr["logit_threshold"],
    )
    return metrics, arrays
