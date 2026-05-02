"""
Normalize CICIDS2017 CSV(s) to canonical flow schema.

CICIDS2017 has ~80 flow features and a Label column (BENIGN / attack type).
We map a subset to our 17 canonical features and set ground_truth = 0 for BENIGN, 1 for attack.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from .schema import CANONICAL_CSV_HEADER, CANONICAL_FEATURES, CANONICAL_LABEL

LOG = logging.getLogger(__name__)

# Map CICIDS2017 column names (various Kaggle uploads) -> canonical feature name
CICIDS_TO_CANONICAL = {
    "Flow Duration": "duration_ms",
    "Flow duration": "duration_ms",
    "FlowDuration": "duration_ms",
    "Total Fwd Packets": "request_sequence",  # use as proxy for activity
    "Total backward packets": "requests_last_60s",
    "Total Length of Fwd Packets": "request_content_length",
    "Total Length of Bwd Packets": "response_size",
    "Fwd Packet Length Mean": "path_length",  # proxy
    "Bwd Packet Length Mean": "response_size",
    "Flow Bytes/s": "header_size_bytes",
    "Flow Packets/s": "requests_last_60s",
    "Flow IAT Mean": "inter_arrival_ms",
    "Flow IAT Std": "query_length",
    "Fwd IAT Mean": "inter_arrival_ms",
    "Fwd IAT Total": "duration_ms",
    "Min flow length": "path_length",
    "Max flow length": "response_size",
    "Subflow Fwd Packets": "request_sequence",
    "Subflow Bwd Packets": "requests_last_60s",
    "Init_Win_bytes_forward": "header_count",
    "Init_Win_bytes_backward": "header_size_bytes",
    "act_data_pkt_fwd": "request_sequence",
    "min_seg_size_forward": "path_length",
    "Active Mean": "duration_ms",
    "Active Std": "query_length",
    "Idle Mean": "inter_arrival_ms",
    "Idle Std": "query_length",
}

# Label column names seen in CICIDS CSVs
LABEL_ALIASES = ("Label", "label", "label ")


def _infer_label_column(headers: list[str]) -> str | None:
    for h in headers:
        if h.strip() in LABEL_ALIASES:
            return h
    return None


def _row_to_ground_truth(row: dict, label_col: str) -> int:
    val = (row.get(label_col) or "").strip()
    if not val or val.upper() == "BENIGN" or val == "0":
        return 0
    return 1


def _normalize_row(
    row: dict,
    cicids_headers: list[str],
    label_col: str,
) -> dict:
    """Build one canonical flow dict from a CICIDS row."""
    out = {f: 0.0 for f in CANONICAL_FEATURES}
    out[CANONICAL_LABEL] = _row_to_ground_truth(row, label_col)

    # Map CICIDS columns (first match wins if multiple map to same canonical)
    for ch in cicids_headers:
        canon = CICIDS_TO_CANONICAL.get(ch.strip())
        if not canon or canon not in CANONICAL_FEATURES:
            continue
        try:
            v = float(row.get(ch, 0) or 0)
        except (ValueError, TypeError):
            v = 0.0
        out[canon] = v

    # Ensure duration_ms in ms (CICIDS often in microseconds)
    if out.get("duration_ms", 0) > 1e6:
        out["duration_ms"] = out["duration_ms"] / 1000.0

    # Placeholders for CSV columns we don't use for training but keep in schema
    out["timestamp"] = str(row.get("Timestamp", row.get("timestamp", "")))
    out["client_ip"] = str(row.get("Source IP", row.get("Source IP", "")))
    out["method"] = ""
    out["path"] = ""
    out["user_agent"] = ""
    out["user_agent_length"] = 0.0
    out["referer"] = ""
    out["referer_present"] = 0.0
    out["accept"] = ""
    out["accept_language"] = ""
    return out


def _find_csvs(folder: Path) -> list[Path]:
    return list(folder.rglob("*.csv"))


def normalize_cicids_folder(folder: Path, output_path: Path, max_rows: int | None = None) -> int:
    """
    Read all CSVs under folder, normalize to canonical schema, write one combined CSV.
    Returns number of rows written.
    """
    folder = Path(folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csvs = _find_csvs(folder)
    if not csvs:
        LOG.warning("No CSV files found under %s", folder)
        return 0

    written = 0
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = None
        for csv_path in sorted(csvs):
            try:
                with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames or []
                    label_col = _infer_label_column(headers)
                    if not label_col:
                        LOG.debug("No label column in %s, skipping", csv_path.name)
                        continue
                    for row in reader:
                        if max_rows is not None and written >= max_rows:
                            break
                        canonical = _normalize_row(row, headers, label_col)
                        if writer is None:
                            writer = csv.DictWriter(out_f, fieldnames=CANONICAL_CSV_HEADER, extrasaction="ignore")
                            writer.writeheader()
                        writer.writerow(canonical)
                        written += 1
            except Exception as e:
                LOG.warning("Error reading %s: %s", csv_path, e)
            if max_rows is not None and written >= max_rows:
                break

    LOG.info("CICIDS2017 normalized: %d rows -> %s", written, output_path)
    return written
