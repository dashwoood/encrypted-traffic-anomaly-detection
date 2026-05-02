"""K-means clustering anomaly detection: distance to centroids."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..base import BaseDetector
from ..registry import register

_DETECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DETECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DETECT_ROOT))

from flow_reader import flows_to_feature_array  # noqa: E402


@register
class KMeansDetector(BaseDetector):
    """K-means: cluster-based anomaly detection (distance to nearest centroid)."""

    name = "kmeans"

    def __init__(self, n_clusters: int = 5, contamination: float = 0.6, random_state: int = 42):
        self.n_clusters = min(n_clusters, 5)
        self.contamination = contamination
        self.random_state = random_state
        self._scaler: StandardScaler | None = None
        self._kmeans: KMeans | None = None
        self._threshold: float = 0.0

    def fit(self, flows: list[dict]) -> None:
        X = flows_to_feature_array(flows)
        n = len(X)
        n_clusters = min(self.n_clusters, max(2, n))
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        self._kmeans.fit(X_scaled)
        distances = self._distances_to_centroids(X_scaled)
        self._threshold = np.percentile(distances, 100 * (1 - self.contamination))

    def _distances_to_centroids(self, X: np.ndarray) -> np.ndarray:
        return np.min(
            np.linalg.norm(X[:, np.newaxis] - self._kmeans.cluster_centers_, axis=2),
            axis=1,
        )

    def predict(self, flows: list[dict]) -> list[bool]:
        if self._kmeans is None:
            self.fit(flows)
        X = flows_to_feature_array(flows)
        X_scaled = self._scaler.transform(X)
        distances = self._distances_to_centroids(X_scaled)
        return [bool(d > self._threshold) for d in distances]

    def predict_scores(self, flows: list[dict]) -> list[float]:
        if self._kmeans is None:
            self.fit(flows)
        X = flows_to_feature_array(flows)
        X_scaled = self._scaler.transform(X)
        distances = self._distances_to_centroids(X_scaled)
        if self._threshold > 0:
            scores = distances / (2.0 * self._threshold)
        else:
            scores = distances
        return [min(float(s), 1.0) for s in scores]

    def save(self, path: Path) -> None:
        if self._kmeans is None:
            raise ValueError("Model not fitted; call fit() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self._scaler,
                "kmeans": self._kmeans,
                "threshold": self._threshold,
                "meta": {
                    "n_clusters": self.n_clusters,
                    "contamination": self.contamination,
                    "random_state": self.random_state,
                },
            },
            path,
        )

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self._scaler = data["scaler"]
        self._kmeans = data["kmeans"]
        self._threshold = data["threshold"]
        meta = data.get("meta", {})
        self.n_clusters = meta.get("n_clusters", 5)
        self.contamination = meta.get("contamination", 0.1)
        self.random_state = meta.get("random_state", 42)
