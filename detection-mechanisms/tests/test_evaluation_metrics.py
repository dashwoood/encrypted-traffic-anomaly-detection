import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import compute_full_metrics  # type: ignore


def test_compute_full_metrics_basic_properties():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_scores = [0.1, 0.8, 0.9, 0.95]

    metrics = compute_full_metrics(y_true, y_pred, y_scores)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["tn"] == 1

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

