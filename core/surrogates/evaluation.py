"""One set of metrics and output arrays for all event models."""

import numpy as np

from core.precision_recall import precision_recall


def evaluate_surrogate(model, target, *, context=None):
    prediction = model.predict(target, context=context)
    labels = np.asarray(target.labels)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Evaluation requires real binary labels, not mixup targets")
    logits = prediction.logits
    if logits.shape != labels.shape:
        raise ValueError("Predictions and labels must have identical shapes")
    observed, predicted = labels.mean(1), prediction.mean
    residual = predicted - observed
    curve = (
        precision_recall(labels.ravel(), logits.ravel())
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
        events=labels.size,
        positives=int(labels.sum()),
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
            np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits)).mean()
        ),
        brier_score=float(np.square(prediction.probabilities - labels).mean()),
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
    return metrics, arrays
