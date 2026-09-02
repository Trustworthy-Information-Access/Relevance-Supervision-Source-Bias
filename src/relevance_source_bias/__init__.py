"""Tools for studying source bias in neural retrieval."""

from .analysis.geometry import estimate_direction
from .interventions.projection import project_out
from .retrieval.metrics import evaluate_run, evaluate_source_preference

__all__ = [
    "estimate_direction",
    "evaluate_run",
    "evaluate_source_preference",
    "project_out",
]

__version__ = "1.0.0"
