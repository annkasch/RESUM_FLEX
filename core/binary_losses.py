"""Numerically stable binary BCE / focal BCE on logits."""
import math
import torch
from torch.nn import functional as F


def binary_focal_loss_with_logits(logits, labels, *, gamma=2.0, weights=None):
    """Mean focal BCE; gamma=0 is BCE. Supports binary or soft targets.

    No additional alpha/class balancing is applied. Optional importance weights
    correct target sampling, and are applied before the ordinary event mean.
    Softplus keeps the corrective gradient for confidently wrong predictions.
    """
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative')
    labels = torch.as_tensor(labels, dtype=logits.dtype, device=logits.device)
    if gamma == 0:
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    else:
        loss = (labels * torch.sigmoid(-logits).pow(gamma) * F.softplus(-logits)
                + (1-labels) * torch.sigmoid(logits).pow(gamma) * F.softplus(logits))
    if weights is not None:
        loss = loss * torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    return loss.mean()
