"""Binary precision/recall at every distinct score, with tied scores grouped."""
import numpy as np


def precision_recall(labels, scores):
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if not len(labels) or labels.shape != scores.shape:
        raise ValueError('Expected matching nonempty label/score arrays')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Expected binary labels and finite scores')
    positives = int(labels.sum())
    if not positives:
        raise ValueError('Recall is undefined without positive observations')
    order = np.argsort(-scores, kind='stable')
    y, score = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(score)), len(score)-1]
    true_positive = np.cumsum(y)[ends]
    precision = true_positive / (ends+1)
    recall = true_positive / positives
    average_precision = float(np.sum(np.diff(np.r_[0., recall]) * precision))
    return dict(precision=np.r_[1., precision], recall=np.r_[0., recall],
                logit_threshold=np.r_[np.inf,score[ends]], average_precision=average_precision,
                prevalence=float(labels.mean()), events=len(labels), positives=positives)
