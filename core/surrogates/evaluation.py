"""One set of metrics and output arrays for all event models."""

from contextlib import contextmanager

import numpy as np

from core.precision_recall import precision_recall

_EVALUATION_THREADS = None


@contextmanager
def evaluation_threads(n: int):
    """Run every evaluate_surrogate call inside the block with ``n`` torch CPU threads."""
    global _EVALUATION_THREADS
    previous, _EVALUATION_THREADS = _EVALUATION_THREADS, n
    try:
        yield
    finally:
        _EVALUATION_THREADS = previous


def _predict(model, target, context):
    if _EVALUATION_THREADS is None:
        return model.predict(target, context=context)
    import torch

    training_threads = torch.get_num_threads()
    torch.set_num_threads(_EVALUATION_THREADS)
    try:
        return model.predict(target, context=context)
    finally:
        torch.set_num_threads(training_threads)


def evaluate_surrogate(model, target, *, context=None):
    prediction = _predict(model, target, context)
    labels = np.asarray(target.labels)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Evaluation requires real binary labels, not mixup targets")
    logits = prediction.logits
    if logits.shape != labels.shape:
        raise ValueError("Predictions and labels must have identical shapes")
    observed, predicted = target.voxel_mean(labels), prediction.mean
    residual = predicted - observed
    # Event-level metrics use only valid events; padded positions are dropped.
    mask = target.event_mask
    event_labels, event_logits = labels[mask], logits[mask]
    curve = (
        precision_recall(event_labels, event_logits)
        if labels.sum()
        else dict(
            average_precision=None,
            prevalence=0.0,
            precision=np.array([]),
            recall=np.array([]),
            logit_threshold=np.array([]),
        )
    )
    mean_observed = float(observed.mean())
    metrics = dict(
        voxels=len(observed),
        events=int(event_labels.size),
        positives=int(event_labels.sum()),
        mean_observed=mean_observed,
        mean_predicted=float(predicted.mean()),
        mean_bias=float(residual.mean()),
        ratio_of_means=float(predicted.mean() / mean_observed) if mean_observed else None,
        pearson_r=float(np.corrcoef(observed, predicted)[0, 1])
        if observed.std() > 0 and predicted.std() > 0
        else None,
        voxel_rate_mae=float(np.abs(residual).mean()),
        voxel_rate_rmse=float(np.sqrt(np.square(residual).mean())),
        bernoulli_log_loss=float(
            np.where(
                event_labels == 1, np.logaddexp(0, -event_logits), np.logaddexp(0, event_logits)
            ).mean()
        ),
        brier_score=float(np.square(prediction.probabilities[mask] - event_labels).mean()),
        average_precision=curve["average_precision"],
        prevalence=curve["prevalence"],
    )
    arrays = dict(
        logits=logits,
        labels=labels,
        observed=observed,
        predicted=predicted,
        residual=residual,
        precision=curve["precision"],
        recall=curve["recall"],
        logit_threshold=curve["logit_threshold"],
    )
    if prediction.legacy_scale is not None:
        arrays["legacy_scale"] = prediction.legacy_scale
    if target.event_counts is not None:
        arrays["event_counts"] = target.event_counts
    return metrics, arrays
