"""Registry of available anomaly detection models."""
from .base import BaseDetector

_REGISTRY: dict[str, type[BaseDetector]] = {}


def register(cls: type[BaseDetector]) -> type[BaseDetector]:
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[BaseDetector]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())
