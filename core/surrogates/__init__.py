"""Interchangeable event models for RESUM_FLEX.

build_surrogate -> fit -> predict -> save/load, independent of model family.
"""

from core.surrogates.base import Episode, EventPrediction, EventSurrogate
from core.surrogates.checkpoints import import_legacy_surrogate, load_surrogate, save_surrogate
from core.surrogates.evaluation import evaluate_surrogate
from core.surrogates.models import build_surrogate
from core.surrogates.pipeline import prepare_surrogate_datasets
from core.surrogates.training import FitResult, fit_surrogate

__all__ = [
    "Episode",
    "EventPrediction",
    "EventSurrogate",
    "FitResult",
    "build_surrogate",
    "evaluate_surrogate",
    "fit_surrogate",
    "load_surrogate",
    "import_legacy_surrogate",
    "prepare_surrogate_datasets",
    "save_surrogate",
]
