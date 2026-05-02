"""Base class for anomaly detection models."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


class BaseDetector(ABC):
    """Abstract base for traditional and AI anomaly detectors."""

    name: str = "base"
    is_sequence_model: bool = False

    @abstractmethod
    def fit(self, flows: list[dict]) -> None:
        """Fit model on normal (or mixed) flow data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, flows: list[dict]) -> list[bool]:
        """Return anomaly flags: True = anomaly, False = normal."""
        raise NotImplementedError

    def predict_scores(self, flows: list[dict]) -> list[float]:
        """Return anomaly scores (higher = more anomalous).

        Default implementation returns 1.0/0.0 from predict().
        Subclasses should override with continuous scores for ROC analysis.
        """
        return [1.0 if p else 0.0 for p in self.predict(flows)]

    def predict_stream(self, flows: Iterator[dict]) -> Iterator[tuple[dict, bool]]:
        """Yield (flow, is_anomaly) for streaming. Default: batch predict."""
        batch = list(flows)
        labels = self.predict(batch)
        for flow, label in zip(batch, labels):
            yield flow, label

    def load(self, path: Path) -> None:
        """Load trained model from path. Override in subclasses."""
        raise NotImplementedError(f"{self.name} does not support load yet")

    def save(self, path: Path) -> None:
        """Save trained model to path. Override in subclasses."""
        raise NotImplementedError(f"{self.name} does not support save yet")
