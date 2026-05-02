"""Ensemble anomaly detection: majority-vote over multiple detectors."""
from __future__ import annotations

from pathlib import Path

import joblib

from ..base import BaseDetector
from ..registry import register, get


@register
class EnsembleDetector(BaseDetector):
    """Ensemble: majority-vote combination of traditional detectors.

    Fits each sub-detector independently and flags a flow as anomalous
    when at least ``ceil(n/2)`` sub-detectors agree.
    """

    name = "ensemble"

    DEFAULT_MEMBERS = ["isolation_forest", "pca", "kmeans"]

    def __init__(self, member_names: list[str] | None = None):
        self.member_names = member_names or list(self.DEFAULT_MEMBERS)
        self._detectors: list[BaseDetector] = []

    def fit(self, flows: list[dict]) -> None:
        self._detectors = []
        for name in self.member_names:
            cls = get(name)
            det = cls()
            det.fit(flows)
            self._detectors.append(det)

    def predict(self, flows: list[dict]) -> list[bool]:
        if not self._detectors:
            self.fit(flows)
        votes = [d.predict(flows) for d in self._detectors]
        majority = len(self._detectors) / 2.0
        return [
            sum(v[i] for v in votes) >= majority
            for i in range(len(flows))
        ]

    def predict_scores(self, flows: list[dict]) -> list[float]:
        if not self._detectors:
            self.fit(flows)
        all_scores = [d.predict_scores(flows) for d in self._detectors]
        n = len(self._detectors)
        return [
            sum(s[i] for s in all_scores) / n
            for i in range(len(flows))
        ]

    def save(self, path: Path) -> None:
        if not self._detectors:
            raise ValueError("Model not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        states = []
        for det in self._detectors:
            sub_path = path.parent / f"_ensemble_{det.name}.joblib"
            det.save(sub_path)
            states.append({"name": det.name, "path": str(sub_path)})
        joblib.dump({"members": states}, path)

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self._detectors = []
        for member in data["members"]:
            cls = get(member["name"])
            det = cls()
            det.load(Path(member["path"]))
            self._detectors.append(det)
        self.member_names = [d.name for d in self._detectors]
