"""Ungrouped events for context-free event models (the neural field), streamed by an
application's event source.

Record layout (the stream format): float32 [M, 7] = 3-D position (x, y, z), 3-D direction
vector (px, py, pz; its length is kept for the stream MLP) and label bits in the last column
(label = any bit set). An event source delivers these records in random order; the run
config names it by import path (``event_source: module:Class``) and passes
``event_source_options`` to its constructor unchanged. RESUM_FLEX never imports a data
package directly.
"""

import importlib
from typing import Protocol


class EventSource(Protocol):
    n_events: int
    n_positive: int  # events with label 1 (training prevalence for the log-loss skill)

    def normalization(self) -> dict: ...  # offset/scale of the 6 physical columns

    def n_batches(self, batch_size: int, pass_index: int, *, drop_last: bool = False) -> int: ...

    def batches(
        self, batch_size: int, pass_index: int = 0, *, start_batch=0, drop_last=False, device="cpu"
    ): ...

    def load(self): ...  # all events of the split as float32 [N, 7]


def event_labels(batch):
    """Labels [M] float of raw records [M, 7] (label bits stored in the last column)."""
    import torch

    return batch[:, 6].contiguous().view(torch.int32).to(batch.dtype)


def open_event_source(spec: str, store, split: str, **options) -> EventSource:
    """Event source ``spec`` ('module:Class') for one split of ``store``."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"event_source must be 'module:Class', got {spec!r}")
    try:
        factory = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import event_source {spec!r}: {exc}") from exc
    return factory(store, split, **options)
