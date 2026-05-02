"""Anomaly detection models."""
from .base import BaseDetector
from .registry import get, list_models, register

# Import subpackages to trigger registration
from . import ai, traditional  # noqa: F401, E402

__all__ = ["BaseDetector", "get", "list_models", "register"]
