"""Visualization module for anomaly detection experiment results.

Generates ROC curves, precision-recall curves, confusion matrices,
and metric comparison bar charts for inclusion in the thesis.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    LOG.warning("matplotlib not installed; visualization disabled")

MODEL_ORDER = [
    "isolation_forest", "pca", "kmeans", "threshold", "ensemble",
    "autoencoder", "cnn", "lstm", "gru", "transformer",
]

MODEL_ORDER_WITH_BASELINE = ["baseline"] + MODEL_ORDER


def _check_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for visualization. pip install matplotlib")


def _comma_fmt(x, _pos):
    """Format axis tick as string with decimal comma."""
    return f"{x:.2f}".replace(".", ",")


def _auc_label(value):
    """Format AUC value with decimal comma for legend."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}".replace(".", ",")
    return str(value)


def _apply_comma_axes(ax, which="both"):
    fmt = FuncFormatter(_comma_fmt)
    if which in ("both", "x"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("both", "y"):
        ax.yaxis.set_major_formatter(fmt)


def _sort_results(results: list[dict], order: list[str]) -> list[dict]:
    """Sort results by MODEL_ORDER; models not in order go at the end."""
    idx_map = {name: i for i, name in enumerate(order)}
    return sorted(results, key=lambda r: idx_map.get(r.get("model", ""), 999))


def plot_roc_curves(results: list[dict], output_dir: Path) -> Path:
    """Overlay ROC curves for all models that have roc_curve data."""
    _check_mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Náhodný klasifikátor")

    for r in _sort_results(results, MODEL_ORDER_WITH_BASELINE):
        rc = r.get("roc_curve")
        if not rc:
            continue
        auc_str = _auc_label(r.get("roc_auc", "?"))
        label = f"{r['model']} (AUC={auc_str})"
        ax.plot(rc["fpr"], rc["tpr"], label=label)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Křivky ROC — všechny modely")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _apply_comma_axes(ax)
    path = output_dir / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOG.info("ROC curves saved to %s", path)
    return path


def plot_pr_curves(results: list[dict], output_dir: Path) -> Path:
    """Overlay precision-recall curves."""
    _check_mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in _sort_results(results, MODEL_ORDER_WITH_BASELINE):
        pc = r.get("pr_curve")
        if not pc:
            continue
        auc_str = _auc_label(r.get("pr_auc", "?"))
        label = f"{r['model']} (PR-AUC={auc_str})"
        ax.plot(pc["recall"], pc["precision"], label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Křivky precision–recall — všechny modely")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _apply_comma_axes(ax)
    path = output_dir / "pr_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOG.info("PR curves saved to %s", path)
    return path


def plot_confusion_matrices(results: list[dict], output_dir: Path) -> Path:
    """Grid of confusion matrices, one per model."""
    _check_mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _sort_results([r for r in results if "tp" in r], MODEL_ORDER_WITH_BASELINE)
    n = len(models)
    if n == 0:
        LOG.warning("No models with confusion matrix data")
        return output_dir / "confusion_matrices.png"

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, r in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normální", "Anomálie"])
        ax.set_yticklabels(["Normální", "Anomálie"])
        ax.set_xlabel("Predikce")
        ax.set_ylabel("Skutečnost")
        ax.set_title(r["model"], fontsize=10)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    path = output_dir / "confusion_matrices.png"
    fig.suptitle("Matice záměn", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOG.info("Confusion matrices saved to %s", path)
    return path


def plot_metric_comparison(results: list[dict], output_dir: Path) -> Path:
    """Grouped bar chart of P/R/F1 for all models (baseline omitted for readability)."""
    _check_mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered = [
        r for r in results
        if "f1" in r and r.get("model") != "baseline"
    ]
    models = _sort_results(filtered, MODEL_ORDER)
    if not models:
        return output_dir / "metric_comparison.png"

    names = [m["model"] for m in models]
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [m.get(metric) or 0.0 for m in models]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Skóre", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_title("Srovnání výkonnosti modelů", fontsize=14, pad=12)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    _apply_comma_axes(ax, which="y")

    path = output_dir / "metric_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Metric comparison saved to %s", path)
    return path


def generate_all_figures(results_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Load results and generate all figures."""
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir / "figures"

    all_results = []
    seen_models: set[str] = set()
    for p in sorted(results_dir.glob("experiment_*_full.json")):
        with open(p) as f:
            data = json.load(f)
        if "error" not in data:
            all_results.append(data)
            seen_models.add(data.get("model", ""))
    for p in sorted(results_dir.glob("experiment_*.json")):
        if "_full" in p.name or p.name == "experiment_summary.json":
            continue
        with open(p) as f:
            data = json.load(f)
        if "error" not in data and data.get("model", "") not in seen_models:
            all_results.append(data)

    if not all_results:
        LOG.warning("No result files found in %s", results_dir)
        return []

    paths = [
        plot_roc_curves(all_results, output_dir),
        plot_pr_curves(all_results, output_dir),
        plot_confusion_matrices(all_results, output_dir),
        plot_metric_comparison(all_results, output_dir),
    ]
    return paths


if __name__ == "__main__":
    import sys
    rdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results")
    paths = generate_all_figures(rdir)
    for p in paths:
        print(p)
