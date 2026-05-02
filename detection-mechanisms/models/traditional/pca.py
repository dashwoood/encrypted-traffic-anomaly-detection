"""PCA-based anomaly detection: reconstruction error threshold."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..base import BaseDetector
from ..registry import register

_DETECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DETECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DETECT_ROOT))

from flow_reader import flows_to_feature_array  # noqa: E402


@register
class PCADetector(BaseDetector):
    """PCA: dimensionality reduction + reconstruction error threshold."""

    name = "pca"

    def __init__(self, n_components: Union[float, int] = 0.95, contamination: float = 0.6):
        self.n_components = n_components
        self.contamination = contamination
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._threshold: float = 0.0

    def fit(self, flows: list[dict]) -> None:
        X = flows_to_feature_array(flows)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(X_scaled)
        recon_errors = self._reconstruction_errors(X_scaled)
        self._threshold = np.percentile(recon_errors, 100 * (1 - self.contamination))

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        X_trans = self._pca.transform(X)
        X_recon = self._pca.inverse_transform(X_trans)
        return np.sum((X - X_recon) ** 2, axis=1)

    def predict(self, flows: list[dict]) -> list[bool]:
        if self._pca is None:
            self.fit(flows)
        X = flows_to_feature_array(flows)
        X_scaled = self._scaler.transform(X)
        errors = self._reconstruction_errors(X_scaled)
        return [bool(e > self._threshold) for e in errors]

    def predict_scores(self, flows: list[dict]) -> list[float]:
        if self._pca is None:
            self.fit(flows)
        X = flows_to_feature_array(flows)
        X_scaled = self._scaler.transform(X)
        errors = self._reconstruction_errors(X_scaled)
        if self._threshold > 0:
            scores = errors / (2.0 * self._threshold)
        else:
            scores = errors
        return [min(float(s), 1.0) for s in scores]

    def save(self, path: Path) -> None:
        if self._pca is None:
            raise ValueError("Model not fitted; call fit() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self._scaler,
                "pca": self._pca,
                "threshold": self._threshold,
                "meta": {"n_components": self.n_components, "contamination": self.contamination},
            },
            path,
        )

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self._scaler = data["scaler"]
        self._pca = data["pca"]
        self._threshold = data["threshold"]
        meta = data.get("meta", {})
        self.n_components = meta.get("n_components", 0.95)
        self.contamination = meta.get("contamination", 0.1)
