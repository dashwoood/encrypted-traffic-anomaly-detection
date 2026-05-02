"""
Prepare train/val from benchmark datasets only (CICIDS2017, UNSW-NB15).

- Optionally downloads benchmarks from Kaggle into data/datasets/
- Normalizes benchmark CSVs to canonical schema
- Writes train/val splits (no synthetic data; benchmarks only)
- Datasets are not stored in git (data/ is gitignored)
"""
from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any

from .schema import CANONICAL_CSV_HEADER, CANONICAL_LABEL
from .download import download_all
from .normalize_cicids import normalize_cicids_folder
from .normalize_unsw import normalize_unsw_folder

LOG = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
DATASETS_DIR = "datasets"
TRAIN_CSV = "canonical_train.csv"
VAL_CSV = "canonical_val.csv"
TRAIN_ORDERED_CSV = "canonical_train_ordered.csv"
VAL_ORDERED_CSV = "canonical_val_ordered.csv"
VAL_RATIO = 0.2
SEED = 42


def _quality_report_for_rows(rows: list[dict]) -> dict[str, Any]:
    """Compute a lightweight quality report for a canonical row list."""
    total = len(rows)
    if not rows:
        return {"total_rows": 0}

    label_counts = {"normal": 0, "anomaly": 0, "missing": 0}
    missing_per_field: dict[str, int] = {k: 0 for k in CANONICAL_CSV_HEADER}

    for row in rows:
        label_val = row.get(CANONICAL_LABEL, "")
        if label_val in (0, "0", 0.0, "normal", "NORMAL"):
            label_counts["normal"] += 1
        elif label_val in (1, "1", 1.0, "anomaly", "ANOMALY"):
            label_counts["anomaly"] += 1
        else:
            label_counts["missing"] += 1

        for key in CANONICAL_CSV_HEADER:
            if key not in row or row[key] in ("", None):
                missing_per_field[key] += 1

    return {
        "total_rows": total,
        "label_counts": label_counts,
        "missing_values": missing_per_field,
    }


def prepare_datasets(
    data_dir: Path,
    *,
    download_benchmarks: bool = True,
    max_cicids_rows: int | None = None,
    max_unsw_rows: int | None = None,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[Path, Path]:
    """
    Prepare canonical_train.csv and canonical_val.csv under data_dir/datasets/.

    Uses benchmark datasets only (CICIDS2017, UNSW-NB15). Synthetic data is not
    included; download_benchmarks must succeed for real evaluation metrics.
    """
    data_dir = Path(data_dir)
    datasets_dir = data_dir / DATASETS_DIR
    datasets_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = datasets_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if download_benchmarks:
        LOG.info("Downloading benchmark datasets from Kaggle...")
        download_all(raw_dir)

    all_rows: list[dict] = []

    cicids_folder = raw_dir / "cicids2017"
    cicids_norm = datasets_dir / "cicids2017_normalized.csv"
    if cicids_folder.exists():
        n = normalize_cicids_folder(cicids_folder, cicids_norm, max_rows=max_cicids_rows)
        if n and cicids_norm.exists():
            with open(cicids_norm, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    all_rows.append(row)
            LOG.info("Added %d CICIDS2017 rows (total %d)", n, len(all_rows))
    else:
        LOG.warning("CICIDS2017 folder not found at %s; run with download_benchmarks=True", cicids_folder)

    unsw_folder = raw_dir / "unsw_nb15"
    unsw_norm = datasets_dir / "unsw_nb15_normalized.csv"
    if unsw_folder.exists():
        n = normalize_unsw_folder(unsw_folder, unsw_norm, max_rows=max_unsw_rows)
        if n and unsw_norm.exists():
            with open(unsw_norm, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    all_rows.append(row)
            LOG.info("Added UNSW-NB15 rows (total %d)", len(all_rows))
    else:
        LOG.warning("UNSW-NB15 folder not found at %s; run with download_benchmarks=True", unsw_folder)

    if not all_rows:
        raise FileNotFoundError(
            "No benchmark data found. Run with download_benchmarks=True and ensure Kaggle "
            "credentials are set (~/.kaggle/kaggle.json). See scripts/setup_kaggle.py."
        )

    n_val_ord = max(1, int(len(all_rows) * val_ratio))
    train_ord = all_rows[: len(all_rows) - n_val_ord]
    val_ord = all_rows[len(all_rows) - n_val_ord :]

    train_ord_path = datasets_dir / TRAIN_ORDERED_CSV
    val_ord_path = datasets_dir / VAL_ORDERED_CSV
    for path, rows in [(train_ord_path, train_ord), (val_ord_path, val_ord)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CANONICAL_CSV_HEADER, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    LOG.info(
        "Prepared ordered train=%s (%d rows) val=%s (%d rows)",
        train_ord_path, len(train_ord), val_ord_path, len(val_ord),
    )

    random.seed(seed)
    random.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * val_ratio))
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    train_path = datasets_dir / TRAIN_CSV
    val_path = datasets_dir / VAL_CSV
    for path, rows in [(train_path, train_rows), (val_path, val_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CANONICAL_CSV_HEADER, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    LOG.info(
        "Prepared shuffled train=%s (%d rows) val=%s (%d rows)",
        train_path,
        len(train_rows),
        val_path,
        len(val_rows),
    )

    for split_name, rows in [
        ("train_shuffled", train_rows),
        ("val_shuffled", val_rows),
        ("train_ordered", train_ord),
        ("val_ordered", val_ord),
    ]:
        report_path = datasets_dir / f"dataset_report_{split_name}.json"
        report = _quality_report_for_rows(rows)
        report["split"] = split_name
        report["data_dir"] = str(data_dir)
        with open(report_path, "w", encoding="utf-8") as f:
            import json

            json.dump(report, f, indent=2)
        LOG.info("Wrote dataset quality report: %s", report_path)

    return train_path, val_path
