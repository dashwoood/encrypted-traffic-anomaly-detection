"""Statistical threshold anomaly detection: z-score based."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

from ..base import BaseDetector
from ..registry import register

_DETECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DETECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DETECT_ROOT))

from flow_reader import flows_to_feature_array  # noqa: E402


@register
class ThresholdDetector(BaseDetector):
    """Statistical threshold: flag flows where any feature exceeds mean +/- k*std.

    Computes per-feature mean and standard deviation on training data and
    flags a flow as anomalous if *any* feature's z-score exceeds *k*.
    """

    name = "threshold"

    def __init__(self, k: float = 3.0):
        self.k = k
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, flows: list[dict]) -> None:
        X = flows_to_feature_array(flows)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0

    def _z_scores(self, X: np.ndarray) -> np.ndarray:
        return np.abs((X - self._mean) / self._std)

    def predict(self, flows: list[dict]) -> list[bool]:
        if self._mean is None:
            self.fit(flows)
        X = flows_to_feature_array(flows)
        z = self._z_scores(X)
        return [bool(row.max() > self.k) for row in z]

    def predict_scores(self, flows: list[dict]) -> list[float]:
        if self._mean is None:
            self.fit(flows)
        X = flows_to_feature_array(flows)
        z = self._z_scores(X)
        max_z = z.max(axis=1)
        scores = max_z / (2.0 * self.k)
        return [min(float(s), 1.0) for s in scores]

    def save(self, path: Path) -> None:
        if self._mean is None:
            raise ValueError("Model not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"mean": self._mean, "std": self._std, "meta": {"k": self.k}}, path)

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self._mean = data["mean"]
        self._std = data["std"]
        self.k = data.get("meta", {}).get("k", 3.0)
