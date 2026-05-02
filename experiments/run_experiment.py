#!/usr/bin/env python3
"""
Run detection experiment: load flows, run detector, evaluate, save results.

Proper evaluation (recommended for thesis):
  python3 run_experiment.py --train data/datasets/canonical_train.csv --test data/datasets/canonical_val.csv --model cnn --output results/
  Fits on --train, predicts on --test, reports precision/recall/F1 on test set.

Run all models:
  python3 run_experiment.py --train ... --test ... --all --output results/

Legacy (in-sample metrics; overfitting risk):
  python3 run_experiment.py --flows data/datasets/canonical_train.csv --model isolation_forest --output results/
"""
import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_ROOT.parent
DETECT_ROOT = PROJECT_ROOT / "detection-mechanisms"
sys.path.insert(0, str(DETECT_ROOT))

from flow_reader import read_flows  # noqa: E402
from models import get, list_models  # noqa: E402
from evaluation import compute_full_metrics  # noqa: E402

TRADITIONAL_MODELS = ["baseline", "isolation_forest", "pca", "kmeans", "threshold", "ensemble"]
AI_MODELS = ["autoencoder", "cnn", "lstm", "gru", "transformer"]
ALL_MODELS = TRADITIONAL_MODELS + AI_MODELS


def run_single(
    train_flows: list[dict],
    train_y: list[int],
    test_flows: list[dict],
    test_y: list[int],
    model_name: str,
    model_dir: Path | None = None,
    window_size: int | None = None,
) -> dict:
    """Fit on train_flows, predict on test_flows, report full metrics on test_y."""
    model_cls = get(model_name)
    detector = model_cls()

    if window_size and hasattr(detector, "window_size"):
        detector.window_size = window_size

    model_path = (model_dir / f"{model_name}.joblib") if model_dir else None
    if model_path and model_path.exists():
        try:
            detector.load(model_path)
        except Exception:
            detector.fit(train_flows)
            _try_save(detector, model_path)
    else:
        detector.fit(train_flows)
        _try_save(detector, model_path)

    y_pred = detector.predict(test_flows)

    y_scores = None
    try:
        y_scores = detector.predict_scores(test_flows)
    except Exception:
        pass

    n_pred = len(y_pred)
    n_true = len(test_y)
    if n_pred < n_true:
        test_y_aligned = test_y[n_true - n_pred:]
    else:
        test_y_aligned = test_y

    metrics = compute_full_metrics(test_y_aligned, y_pred, y_scores)
    metrics["model"] = model_name
    if hasattr(detector, "is_sequence_model") and detector.is_sequence_model:
        metrics["window_size"] = getattr(detector, "window_size", None)
    return metrics


def _try_save(detector, model_path):
    if model_path and hasattr(detector, "save"):
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            detector.save(model_path)
        except (NotImplementedError, Exception):
            pass


def _read_flows_and_labels(path: Path) -> tuple[list[dict], list[int]]:
    flows_list = list(read_flows(path))
    if not flows_list:
        return [], []
    gt_col = "ground_truth" if "ground_truth" in flows_list[0] else "anomaly"
    if gt_col not in flows_list[0]:
        raise ValueError("flows CSV missing ground_truth/anomaly column.")

    def _gt(f: dict) -> int:
        v = f.get(gt_col, f.get("ground_truth", 0))
        try:
            return 1 if int(float(v or 0)) else 0
        except (ValueError, TypeError):
            return 0

    return flows_list, [_gt(f) for f in flows_list]


def _run_cross_validation_for_model(
    flows: list[dict],
    y: list[int],
    model_name: str,
    window_size: int,
    cv_folds: int,
) -> dict:
    """Run k-fold cross-validation for a single model on (flows, y).

    Uses StratifiedKFold on the provided labels to produce *cv_folds* splits.
    For each split, the model is trained on the training portion and evaluated
    on the held-out fold. Models are not persisted during cross-validation.
    """
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2 for cross-validation")

    model_cls = get(model_name)
    if getattr(model_cls, "is_sequence_model", False):
        return _run_block_cv_for_sequence(
            flows, y, model_name, window_size=window_size, cv_folds=cv_folds
        )

    flows_arr = np.array(flows, dtype=object)
    y_arr = np.array(y, dtype=int)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_results: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(flows_arr, y_arr), start=1):
        train_flows = flows_arr[train_idx].tolist()
        train_y = y_arr[train_idx].tolist()
        val_flows = flows_arr[val_idx].tolist()
        val_y = y_arr[val_idx].tolist()

        metrics = run_single(
            train_flows,
            train_y,
            val_flows,
            val_y,
            model_name,
            model_dir=None,
            window_size=window_size,
        )
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

    f1_scores = [m["f1"] for m in fold_results if "f1" in m]
    prec_scores = [m["precision"] for m in fold_results if "precision" in m]
    rec_scores = [m["recall"] for m in fold_results if "recall" in m]

    summary: dict = {
        "model": model_name,
        "cv_folds": cv_folds,
        "folds": fold_results,
    }
    if f1_scores:
        summary["f1_mean"] = float(np.mean(f1_scores))
        summary["f1_std"] = float(np.std(f1_scores))
    if prec_scores:
        summary["precision_mean"] = float(np.mean(prec_scores))
    if rec_scores:
        summary["recall_mean"] = float(np.mean(rec_scores))
    return summary


def _run_block_cv_for_sequence(
    flows: list[dict],
    y: list[int],
    model_name: str,
    window_size: int,
    cv_folds: int,
) -> dict:
    """Contiguous k-fold on time-ordered flows (train = all segments except held-out)."""
    n = len(flows)
    if n < cv_folds + window_size + 5:
        return {
            "model": model_name,
            "note": "Insufficient rows for block cross-validation on sequence model.",
        }

    boundaries: list[int] = [0]
    base = n // cv_folds
    rem = n % cv_folds
    pos = 0
    for i in range(cv_folds):
        seg_len = base + (1 if i < rem else 0)
        pos += seg_len
        boundaries.append(pos)

    fold_results: list[dict] = []
    for k in range(cv_folds):
        vs, ve = boundaries[k], boundaries[k + 1]
        if ve - vs < window_size + 2:
            continue
        train_flows = flows[:vs] + flows[ve:]
        train_y = y[:vs] + y[ve:]
        val_flows = flows[vs:ve]
        val_y = y[vs:ve]
        if len(train_flows) < window_size + 2:
            continue
        metrics = run_single(
            train_flows,
            train_y,
            val_flows,
            val_y,
            model_name,
            model_dir=None,
            window_size=window_size,
        )
        metrics["fold"] = k + 1
        fold_results.append(metrics)

    f1_scores = [m["f1"] for m in fold_results if "f1" in m]
    summary: dict = {
        "model": model_name,
        "cv_folds": cv_folds,
        "cv_scheme": "contiguous_block_sequence",
        "folds": fold_results,
    }
    if f1_scores:
        summary["f1_mean"] = float(np.mean(f1_scores))
        summary["f1_std"] = float(np.std(f1_scores))
    if not fold_results:
        summary["note"] = "No valid folds produced for sequence model (check data size)."
    return summary


def _compute_file_info(path: Path | None) -> dict[str, Any] | None:
    """Return basic metadata and checksum for a given file, if it exists."""
    if path is None:
        return None
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        return None

    if not resolved.exists() or not resolved.is_file():
        return None

    import hashlib

    h = hashlib.sha256()
    size = 0
    try:
        with resolved.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                size += len(chunk)
                h.update(chunk)
    except OSError:
        return {
            "path": str(resolved),
            "exists": True,
            "size_bytes": size if size else None,
            "checksum_sha256": None,
            "error": "failed_to_read_for_checksum",
        }

    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": size,
        "checksum_sha256": h.hexdigest(),
    }


def _write_run_manifest(
    manifest_path: Path,
    *,
    args: argparse.Namespace,
    models_to_run: list[str],
    start_time: datetime,
    end_time: datetime,
    results_dir: Path,
) -> None:
    """Persist a machine-readable manifest describing this experiment run."""
    results_dir = results_dir.resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": end_time.replace(tzinfo=timezone.utc).isoformat(),
        "started_at_utc": start_time.replace(tzinfo=timezone.utc).isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "command": " ".join(sys.argv),
        "project_root": str(EXP_ROOT.parent.resolve()),
        "results_dir": str(results_dir),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "environment": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
        "data": {
            "train": _compute_file_info(args.train),
            "test": _compute_file_info(args.test),
            "train_ordered": _compute_file_info(args.train_ordered),
            "test_ordered": _compute_file_info(args.test_ordered),
        },
        "run_config": {
            "seed": getattr(args, "seed", None),
            "max_train_samples": args.max_train_samples,
            "window_size": args.window_size,
            "cv_folds": args.cv_folds,
            "models": models_to_run,
            "all": args.all,
            "traditional": args.traditional,
            "ai": args.ai,
        },
    }

    artifacts: dict[str, Any] = {}
    if results_dir.exists():
        experiment_files = sorted(results_dir.glob("experiment_*.json"))
        cv_files = sorted(results_dir.glob("cv_*.json"))
        summary_files = [
            p for p in results_dir.glob("*.json") if p.name.endswith("summary.json")
        ]
        stat_file = results_dir / "statistical_comparison.json"
        run_manifest_existing = results_dir / "run_manifest.json"

        artifacts["experiment_files"] = [f.name for f in experiment_files]
        artifacts["cv_files"] = [f.name for f in cv_files]
        artifacts["summary_files"] = [f.name for f in summary_files]
        artifacts["statistical_comparison"] = stat_file.name if stat_file.exists() else None
        artifacts["prior_run_manifest"] = (
            str(run_manifest_existing)
            if run_manifest_existing.exists() and run_manifest_existing != manifest_path
            else None
        )

        figures_dir = results_dir / "figures"
        if figures_dir.exists() and figures_dir.is_dir():
            artifacts["figures"] = sorted(
                str(p.relative_to(results_dir)) for p in figures_dir.glob("*")
            )
        else:
            artifacts["figures"] = []

    manifest["artifacts"] = artifacts

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def _validate_results_completeness(
    results_dir: Path,
    models_to_run: list[str],
) -> dict[str, Any]:
    """Check that per-model and summary artifacts are present for all models.

    Returns a dict with any issues found; empty dict means no problems.
    """
    issues: dict[str, Any] = {}
    results_dir = results_dir.resolve()

    missing_files = [
        m for m in models_to_run if not (results_dir / f"experiment_{m}.json").exists()
    ]
    if missing_files:
        issues["missing_experiment_files"] = missing_files

    summary_path = results_dir / "experiment_summary.json"
    if summary_path.exists():
        try:
            with summary_path.open() as f:
                summary_data = json.load(f)
            present_models = {
                entry.get("model")
                for entry in summary_data
                if isinstance(entry, dict) and "model" in entry
            }
            missing_in_summary = [m for m in models_to_run if m not in present_models]
            if missing_in_summary:
                issues["missing_in_summary"] = missing_in_summary
        except Exception as e:
            issues["summary_read_error"] = str(e)
    else:
        issues["summary_missing"] = True

    return issues


def _summarize_ai_benefit_conditions(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce a coarse summary of where AI models outperform traditional ones.

    This operates on the in-memory metrics list (all_results) and is intended
    for downstream thesis analysis. It does not change evaluation itself.
    """
    from math import isfinite

    ai_models = {m for m in ALL_MODELS if m in AI_MODELS}
    trad_models = {m for m in ALL_MODELS if m in TRADITIONAL_MODELS}

    ai = [r for r in results if r.get("model") in ai_models and "f1" in r]
    trad = [r for r in results if r.get("model") in trad_models and "f1" in r]
    if not ai or not trad:
        return {"note": "Insufficient AI/traditional metrics for comparison."}

    def _safe(v: Any) -> float:
        try:
            f = float(v)
        except Exception:
            return 0.0
        return f if isfinite(f) else 0.0

    best_ai = max(ai, key=lambda r: _safe(r.get("f1", 0.0)))
    best_trad = max(trad, key=lambda r: _safe(r.get("f1", 0.0)))

    summary: dict[str, Any] = {
        "best_ai_model": best_ai.get("model"),
        "best_ai_f1": _safe(best_ai.get("f1", 0.0)),
        "best_traditional_model": best_trad.get("model"),
        "best_traditional_f1": _safe(best_trad.get("f1", 0.0)),
    }

    if summary["best_ai_f1"] or summary["best_traditional_f1"]:
        summary["delta_f1_best_ai_minus_best_traditional"] = (
            summary["best_ai_f1"] - summary["best_traditional_f1"]
        )

    # Simple recall-priority view
    ai_recalls = [_safe(r.get("recall", 0.0)) for r in ai]
    trad_recalls = [_safe(r.get("recall", 0.0)) for r in trad]
    if ai_recalls and trad_recalls:
        summary["mean_recall_ai"] = float(np.mean(ai_recalls))
        summary["mean_recall_traditional"] = float(np.mean(trad_recalls))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", type=Path, default=None,
                        help="Path to flows.csv (legacy: fit and eval on same file)")
    parser.add_argument("--train", type=Path, default=None,
                        help="Path to training flows CSV (recommended with --test)")
    parser.add_argument("--test", type=Path, default=None,
                        help="Path to test flows CSV for evaluation")
    parser.add_argument(
        "--train-ordered",
        type=Path,
        default=None,
        help="Optional: training flows CSV with preserved temporal order for sequence models "
             "(e.g. canonical_train_ordered.csv). Per-flow models ignore this.",
    )
    parser.add_argument(
        "--test-ordered",
        type=Path,
        default=None,
        help="Optional: test flows CSV with preserved temporal order for sequence models "
             "(e.g. canonical_val_ordered.csv). Per-flow models ignore this.",
    )
    parser.add_argument("--model", "-m", help="Model name (ignored if --all)")
    parser.add_argument("--all", action="store_true",
                        help="Run all available models (traditional + AI)")
    parser.add_argument("--traditional", action="store_true",
                        help="Run traditional models only")
    parser.add_argument("--ai", action="store_true", help="Run AI models only")
    parser.add_argument("--output", "-o", type=Path,
                        default=EXP_ROOT / "results", help="Output dir")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Load/save model weights (skip refit if present)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Subsample training data (for faster iteration)")
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Window size for sequence models (LSTM/GRU/Transformer)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and model behavior.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Optional path to save a JSON manifest describing this run "
        "(defaults to OUTPUT/run_manifest.json).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If > 1, run k-fold cross-validation on the training set for each selected model.",
    )
    args = parser.parse_args()

    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    run_start = datetime.now(timezone.utc)
    exit_code = 0

    try:
        if args.train is not None and args.test is not None:
            train_flows, train_y = _read_flows_and_labels(args.train)
            test_flows, test_y = _read_flows_and_labels(args.test)
            if args.max_train_samples and len(train_flows) > args.max_train_samples:
                import random
                random.seed(42)
                idx = random.sample(range(len(train_flows)), args.max_train_samples)
                train_flows = [train_flows[i] for i in idx]
                train_y = [train_y[i] for i in idx]
                print(f"Subsampled to {len(train_flows)} train samples", file=sys.stderr)
            if not train_flows:
                print("No training flows found.", file=sys.stderr)
                return 1
            if not test_flows:
                print("No test flows found.", file=sys.stderr)
                return 1
            flows_file_label = f"train={args.train} test={args.test}"
        elif args.flows is not None:
            train_flows, train_y = _read_flows_and_labels(args.flows)
            if not train_flows:
                print("No flows found.", file=sys.stderr)
                return 1
            test_flows, test_y = train_flows, train_y
            flows_file_label = str(args.flows)
            print("Warning: using same file for fit and evaluation (in-sample metrics).",
                  file=sys.stderr)
        else:
            print("Provide --flows (legacy) or both --train and --test.", file=sys.stderr)
            return 1
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    # Optional: separate ordered datasets for sequence models
    train_flows_seq = train_y_seq = test_flows_seq = test_y_seq = None
    flows_file_label_seq = None
    if args.train_ordered is not None and args.test_ordered is not None:
        train_flows_seq, train_y_seq = _read_flows_and_labels(args.train_ordered)
        test_flows_seq, test_y_seq = _read_flows_and_labels(args.test_ordered)
        if not train_flows_seq or not test_flows_seq:
            print("Ordered train/test provided for sequence models but no flows found.",
                  file=sys.stderr)
            return 1
        flows_file_label_seq = f"train={args.train_ordered} test={args.test_ordered}"
    else:
        # Fallback: use the same shuffled datasets for sequence models if ordered ones are absent.
        train_flows_seq, train_y_seq = train_flows, train_y
        test_flows_seq, test_y_seq = test_flows, test_y
        flows_file_label_seq = flows_file_label

    available = list_models()
    if args.all:
        models_to_run = [m for m in ALL_MODELS if m in available]
    elif args.traditional:
        models_to_run = [m for m in TRADITIONAL_MODELS if m in available]
    elif args.ai:
        models_to_run = [m for m in AI_MODELS if m in available]
    else:
        model = args.model or "isolation_forest"
        models_to_run = [model]

    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_name in models_to_run:
        try:
            model_cls = get(model_name)
            use_seq = getattr(model_cls, "is_sequence_model", False)
            if use_seq:
                tf, ty = train_flows_seq, train_y_seq
                tfl, tyl = test_flows_seq, test_y_seq
                flows_label = flows_file_label_seq
            else:
                tf, ty = train_flows, train_y
                tfl, tyl = test_flows, test_y
                flows_label = flows_file_label

            metrics = run_single(
                tf, ty, tfl, tyl,
                model_name, args.model_dir,
                window_size=args.window_size,
            )
            metrics["flows_file"] = flows_label

            serializable = {k: v for k, v in metrics.items()
                           if k not in ("roc_curve", "pr_curve")}
            out_file = args.output / f"experiment_{model_name}.json"
            with open(out_file, "w") as f:
                json.dump(serializable, f, indent=2)

            full_file = args.output / f"experiment_{model_name}_full.json"
            with open(full_file, "w") as f:
                json.dump(metrics, f, indent=2, default=_json_default)

            all_results.append(metrics)
            roc = metrics.get("roc_auc", "-")
            print(f"{model_name}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                  f"F1={metrics['f1']:.4f} AUC={roc} -> {out_file}")
        except Exception as e:
            print(f"{model_name}: FAILED - {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            all_results.append({"model": model_name, "error": str(e)})

    if len(all_results) > 1:
        summary = [{k: v for k, v in r.items() if k not in ("roc_curve", "pr_curve")}
                   for r in all_results]
        summary_file = args.output / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_file}")

        try:
            ai_benefit = _summarize_ai_benefit_conditions(all_results)
            benefit_path = args.output / "ai_benefit_summary.json"
            with benefit_path.open("w") as f:
                json.dump(ai_benefit, f, indent=2, default=_json_default)
            print(f"AI benefit summary saved to {benefit_path}")
        except Exception as e:
            print(f"AI benefit summary generation failed: {e}", file=sys.stderr)

    try:
        from visualization import generate_all_figures
        fig_dir = args.output / "figures"
        paths = generate_all_figures(args.output, fig_dir)
        if paths:
            print(f"Figures saved to {fig_dir}/")
    except ImportError:
        print("(matplotlib not available; skipping figure generation)", file=sys.stderr)
    except Exception as e:
        print(f"Figure generation failed: {e}", file=sys.stderr)

    # k-fold cross-validation (before statistical tests so Wilcoxon can read cv_*.json)
    try:
        if args.cv_folds and args.cv_folds > 1 and train_flows and train_y:
            print(f"\nRunning {args.cv_folds}-fold cross-validation on training data...",
                  file=sys.stderr)
            cv_summaries: list[dict] = []
            for model_name in models_to_run:
                try:
                    model_cls = get(model_name)
                    use_seq = getattr(model_cls, "is_sequence_model", False)
                    cv_train_f, cv_train_y = (
                        (train_flows_seq, train_y_seq)
                        if use_seq and train_flows_seq is not None
                        else (train_flows, train_y)
                    )
                    summary = _run_cross_validation_for_model(
                        cv_train_f,
                        cv_train_y,
                        model_name,
                        window_size=args.window_size,
                        cv_folds=args.cv_folds,
                    )
                    cv_summaries.append(summary)
                    cv_path = args.output / f"cv_{model_name}.json"
                    with open(cv_path, "w") as f:
                        json.dump(summary, f, indent=2, default=_json_default)
                    print(f"CV results for {model_name} saved to {cv_path}", file=sys.stderr)
                except Exception as cv_err:
                    print(f"Cross-validation for {model_name} failed: {cv_err}",
                          file=sys.stderr)

            if cv_summaries:
                cv_summary_path = args.output / "cv_summary.json"
                with open(cv_summary_path, "w") as f:
                    json.dump(cv_summaries, f, indent=2, default=_json_default)
                print(f"Cross-validation summary saved to {cv_summary_path}",
                      file=sys.stderr)
    except Exception as e:
        print(f"Cross-validation block failed: {e}", file=sys.stderr)

    # Run statistical tests (after CV so paired Wilcoxon has fold-level F1)
    try:
        from statistical_tests import run_comparison
        comparison = run_comparison(args.output)
        comp_file = args.output / "statistical_comparison.json"
        with open(comp_file, "w") as f:
            json.dump(comparison, f, indent=2, default=_json_default)
        print(f"Statistical comparison saved to {comp_file}")
    except Exception as e:
        print(f"Statistical comparison failed: {e}", file=sys.stderr)
        # Mark as an error for completeness enforcement when multiple models were requested.
        if len(models_to_run) > 1:
            exit_code = 1

    try:
        completeness_issues = _validate_results_completeness(args.output, models_to_run)
        if completeness_issues:
            exit_code = 1
            print(
                "Result completeness check reported issues: "
                f"{json.dumps(completeness_issues, indent=2)}",
                file=sys.stderr,
            )
    except Exception as e:
        exit_code = 1
        print(f"Result completeness validation failed: {e}", file=sys.stderr)

    run_end = datetime.now(timezone.utc)

    try:
        manifest_path = args.run_manifest or (args.output / "run_manifest.json")
        _write_run_manifest(
            manifest_path,
            args=args,
            models_to_run=models_to_run,
            start_time=run_start,
            end_time=run_end,
            results_dir=args.output,
        )
        print(f"Run manifest saved to {manifest_path}")
    except Exception as e:
        exit_code = 1
        print(f"Run manifest generation failed: {e}", file=sys.stderr)

    return exit_code


def _json_default(obj):
    """JSON serialization fallback for numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    sys.exit(main())
