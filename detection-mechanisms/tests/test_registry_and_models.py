import sys
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_reader import FEATURE_COLUMNS, flows_to_feature_array  # type: ignore
from models import get, list_models  # type: ignore


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _make_dummy_flows(n: int = 20) -> list[dict]:
    flows: list[dict] = []
    for i in range(n):
        flow: dict = {}
        for col in FEATURE_COLUMNS:
            flow[col] = float(i % 5)
        flow["ground_truth"] = 1.0 if i % 2 else 0.0
        flows.append(flow)
    return flows


def test_model_registry_contains_expected_models():
    models = set(list_models())
    for name in {"baseline", "isolation_forest", "pca", "kmeans", "threshold", "ensemble"}:
        assert name in models
    if _has_torch():
        assert "cnn" in models


def test_flows_to_feature_array_shape():
    flows = _make_dummy_flows(10)
    X = flows_to_feature_array(flows)
    assert isinstance(X, np.ndarray)
    assert X.shape == (10, len(FEATURE_COLUMNS))


def test_traditional_models_fit_and_predict_on_dummy_flows():
    flows = _make_dummy_flows(30)
    for name in ["isolation_forest", "pca", "kmeans", "threshold", "ensemble"]:
        cls = get(name)
        det = cls()
        det.fit(flows)
        preds = det.predict(flows)
        assert len(preds) == len(flows)

