"""Isolation Forest anomaly detection."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from ..base import BaseDetector
from ..registry import register

_DETECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DETECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DETECT_ROOT))

from flow_reader import flows_to_feature_array  # noqa: E402


@register
class IsolationForestDetector(BaseDetector):
    """Isolation Forest: tree-based anomaly detection."""

    name = "isolation_forest"

    def __init__(self, contamination: float = 0.5, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self._model: IsolationForest | None = None

    def fit(self, flows: list[dict]) -> None:
        X = flows_to_feature_array(flows)
        self._model = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self._model.fit(X)

    def predict(self, flows: list[dict]) -> list[bool]:
        if self._model is None:
            self.fit(flows)
        X = flows_to_feature_array(flows)
        pred = self._model.predict(X)
        return [p == -1 for p in pred]

    def predict_scores(self, flows: list[dict]) -> list[float]:
        if self._model is None:
            self.fit(flows)
        X = flows_to_feature_array(flows)
        raw = self._model.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(raw))
        return scores.tolist()

    def save(self, path: Path) -> None:
        if self._model is None:
            raise ValueError("Model not fitted; call fit() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"contamination": self.contamination, "random_state": self.random_state}
        joblib.dump({"model": self._model, "meta": meta}, path)

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self._model = data["model"]
        self.contamination = data.get("meta", {}).get("contamination", 0.1)
        self.random_state = data.get("meta", {}).get("random_state", 42)
