"""Shared configuration and data primitives."""

from .config import load_yaml
from .data import EmbeddingTable

__all__ = ["EmbeddingTable", "load_yaml"]
