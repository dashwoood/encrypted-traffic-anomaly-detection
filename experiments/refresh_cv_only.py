#!/usr/bin/env python3
"""Regenerate cv_<model>.json for all models without rerunning main experiment_* metrics.

Reads training CSVs, runs k-fold CV per model (ordered train for sequence models),
writes stripped cv_*.json, then rebuilds statistical_comparison.json from existing
experiment_*.json plus the new CV files.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "detection-mechanisms"))
sys.path.insert(0, str(ROOT / "experiments"))

import run_experiment as rex  # noqa: E402
from models.registry import get as get_model  # noqa: E402
from statistical_tests import run_comparison  # noqa: E402


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def strip_large_fields(obj: dict) -> None:
    for k in list(obj.keys()):
        if k in ("roc_curve", "pr_curve"):
            del obj[k]
        elif isinstance(obj[k], dict):
            strip_large_fields(obj[k])
        elif isinstance(obj[k], list) and obj[k] and isinstance(obj[k][0], dict):
            for item in obj[k]:
                if isinstance(item, dict):
                    strip_large_fields(item)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv-folds", type=int, default=10)
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training rows (random subsample for per-flow; "
        "prefix for ordered). For full-thesis parity omit this.",
    )
    ap.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "results")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    data = ROOT / "data" / "datasets"
    train_flows, train_y = rex._read_flows_and_labels(data / "canonical_train.csv")
    train_ord, train_y_ord = rex._read_flows_and_labels(data / "canonical_train_ordered.csv")

    if args.max_train_samples is not None:
        n = args.max_train_samples
        if len(train_flows) > n:
            rng = random.Random(42)
            idx = rng.sample(range(len(train_flows)), n)
            train_flows = [train_flows[i] for i in sorted(idx)]
            train_y = [train_y[i] for i in sorted(idx)]
        if len(train_ord) > n:
            rng = random.Random(42)
            pick = sorted(rng.sample(range(len(train_ord)), n))
            train_ord = [train_ord[i] for i in pick]
            train_y_ord = [train_y_ord[i] for i in pick]

    models = args.models if args.models else rex.ALL_MODELS
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cv_summaries: list[dict] = []

    for model_name in models:
        model_cls = get_model(model_name)
        use_seq = getattr(model_cls, "is_sequence_model", False)
        tf, ty = (train_ord, train_y_ord) if use_seq else (train_flows, train_y)
        summary = rex._run_cross_validation_for_model(
            tf, ty, model_name, args.window_size, args.cv_folds
        )
        strip_large_fields(summary)
        cv_summaries.append(summary)
        out_path = args.results_dir / f"cv_{model_name}.json"
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        print(f"Wrote {out_path}", file=sys.stderr)

    cv_summary_path = args.results_dir / "cv_summary.json"
    with cv_summary_path.open("w") as f:
        json.dump(cv_summaries, f, indent=2, default=_json_default)

    comp = run_comparison(args.results_dir)
    for m in comp.get("per_model", {}).values():
        if isinstance(m, dict):
            m.pop("roc_curve", None)
            m.pop("pr_curve", None)

    comp_path = args.results_dir / "statistical_comparison.json"
    with comp_path.open("w") as f:
        json.dump(comp, f, indent=2, default=_json_default)
    print(f"Wrote {comp_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
