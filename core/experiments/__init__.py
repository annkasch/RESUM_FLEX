"""Controlled experiments, immutable artifacts and shared reporting.

Public operations are lazy so catalog access does not import optional model backends.
"""

from importlib import import_module

__all__ = ["run", "validate_experiment", "compare", "compare_saved", "regenerate", "RunStore"]


def __getattr__(name):
    modules = {
        "run": "runner",
        "validate_experiment": "runner",
        "compare": "comparison",
        "compare_saved": "comparison",
        "regenerate": "reporting",
        "RunStore": "storage",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"core.experiments.{modules[name]}"), name)
