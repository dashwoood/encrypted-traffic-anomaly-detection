import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "model_name",
    ["baseline", "isolation_forest", "pca", "kmeans", "threshold", "ensemble", "autoencoder", "cnn"],
)
def test_experiment_file_schema_basic(model_name: str) -> None:
    """Basic schema sanity for experiment_<model>.json files produced by experiments."""
    results_dir = Path(__file__).resolve().parents[2] / "experiments" / "results"
    path = results_dir / f"experiment_{model_name}.json"
    if not path.exists():
        pytest.skip(f"{path} not present – run evaluate-all before this test.")

    with path.open() as f:
        data = json.load(f)

    for key in ("model", "precision", "recall", "f1", "accuracy"):
        assert key in data

    assert data["model"] == model_name
    assert 0.0 <= float(data["precision"]) <= 1.0
    assert 0.0 <= float(data["recall"]) <= 1.0
    assert 0.0 <= float(data["f1"]) <= 1.0
    assert 0.0 <= float(data["accuracy"]) <= 1.0


def test_experiment_summary_covers_all_present_models() -> None:
    """experiment_summary.json, if present, must include an entry per model file."""
    results_dir = Path(__file__).resolve().parents[2] / "experiments" / "results"
    summary_path = results_dir / "experiment_summary.json"
    if not summary_path.exists():
        pytest.skip("experiment_summary.json not present – run evaluate-all before this test.")

    with summary_path.open() as f:
        summary = json.load(f)

    models_in_summary = {
        entry.get("model") for entry in summary if isinstance(entry, dict) and "model" in entry
    }

    for exp_path in results_dir.glob("experiment_*.json"):
        if exp_path.name.endswith("_full.json"):
            continue
        if exp_path.name == "experiment_summary.json":
            continue
        with exp_path.open() as f:
            data = json.load(f)
        model_name = data.get("model")
        assert model_name in models_in_summary


def test_statistical_comparison_schema_if_present() -> None:
    """statistical_comparison.json, if present, should at least contain per_model dict."""
    results_dir = Path(__file__).resolve().parents[2] / "experiments" / "results"
    comp_path = results_dir / "statistical_comparison.json"
    if not comp_path.exists():
        pytest.skip("statistical_comparison.json not present – run evaluate-all before this test.")

    with comp_path.open() as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "per_model" in data
    assert isinstance(data["per_model"], dict)

