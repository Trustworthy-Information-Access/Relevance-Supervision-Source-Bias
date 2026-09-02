"""Dense and lexical retrieval, model adapters, and evaluation metrics."""

from .dense import ModelSpec, exact_search, load_model_registry, retrieve_dataset
from .metrics import evaluate_run, evaluate_source_preference

__all__ = [
    "ModelSpec",
    "evaluate_run",
    "evaluate_source_preference",
    "exact_search",
    "load_model_registry",
    "retrieve_dataset",
]
