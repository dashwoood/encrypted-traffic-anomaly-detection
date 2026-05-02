"""Baseline detector: all flows marked normal. Used for pipeline testing."""
from ..base import BaseDetector
from ..registry import register


@register
class BaselineDetector(BaseDetector):
    """Baseline: marks all flows as normal. For testing the detection pipeline."""

    name = "baseline"

    def fit(self, flows: list[dict]) -> None:
        pass

    def predict(self, flows: list[dict]) -> list[bool]:
        return [False] * len(flows)
