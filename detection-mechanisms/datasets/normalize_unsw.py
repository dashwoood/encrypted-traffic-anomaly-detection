"""
Normalize UNSW-NB15 CSV(s) to canonical flow schema.

UNSW-NB15 has 49 features and label (0=normal, 1=attack) or attack_cat.
We map overlapping semantics to our 17 canonical features.

**Mapping quality note**: The canonical schema was designed for HTTP flow
metadata (e.g. ``path_length``, ``user_agent_length``).  UNSW-NB15 records
low-level network statistics (TTL, jitter, TCP window sizes) that have no
direct HTTP counterpart.  The mapping below selects *one* UNSW column per
canonical field using the closest semantic match available.  Where no
reasonable match exists the canonical field stays at its default (0).
Duplicate/overwriting mappings from the earlier version have been removed;
see inline comments for rationale.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from .schema import CANONICAL_CSV_HEADER, CANONICAL_FEATURES, CANONICAL_LABEL

LOG = logging.getLogger(__name__)

# Each canonical field is mapped to at most *one* UNSW column.
# Rationale is given inline.  Fields with no good match are omitted and
# default to 0 in the output row.
UNSW_TO_CANONICAL = {
    # dur (seconds) -> duration_ms (approximate; unit mismatch noted)
    "dur": "duration_ms",
    # sbytes (source->dest bytes) -> request_content_length
    "sbytes": "request_content_length",
    # dbytes (dest->source bytes) -> response_size
    "dbytes": "response_size",
    # sttl (source TTL, 0-255) -> header_count (both are small integers
    #   describing connection metadata; approximate)
    "sttl": "header_count",
    # dttl (dest TTL) -> header_size_bytes (approximate, small integer)
    "dttl": "header_size_bytes",
    # spkts (source packet count) -> request_sequence
    "spkts": "request_sequence",
    # dpkts (dest packet count) -> requests_last_60s
    "dpkts": "requests_last_60s",
    # smean (mean source packet size) -> path_length (size-like feature)
    "smean": "path_length",
    # dmean (mean dest packet size) -> query_length (size-like feature)
    "dmean": "query_length",
    # ct_srv_src (connection count to same service+src) -> unique_paths_count
    "ct_srv_src": "unique_paths_count",
    # sinpkt (source inter-packet arrival time) -> inter_arrival_ms
    "sinpkt": "inter_arrival_ms",
    # ct_dst_ltm (connections to same dest in last 100 records) -> minute
    #   (temporal activity proxy)
    "ct_dst_ltm": "minute",
    # ct_src_ltm (connections from same src in last 100 records) -> hour_utc
    #   (temporal activity proxy)
    "ct_src_ltm": "hour_utc",
}

# Label: UNSW uses 'label' (0/1) or 'attack_cat' (string)
LABEL_ALIASES = ("label", "Label", "attack_cat")


def _infer_label_column(headers: list[str]) -> str | None:
    for h in headers:
        hc = h.strip().lower()
        if hc in ("label", "attack_cat"):
            return h
    return None


def _row_to_ground_truth(row: dict, label_col: str) -> int:
    val = row.get(label_col)
    if val is None or val == "":
        return 0
    if isinstance(val, (int, float)):
        return 1 if int(val) != 0 else 0
    return 0 if str(val).strip().lower() in ("normal", "0", "") else 1


def _normalize_row(row: dict, unsw_headers: list[str], label_col: str) -> dict:
    out = {f: 0.0 for f in CANONICAL_FEATURES}
    out[CANONICAL_LABEL] = _row_to_ground_truth(row, label_col)

    for h in unsw_headers:
        canon = UNSW_TO_CANONICAL.get(h.strip())
        if not canon or canon not in CANONICAL_FEATURES:
            continue
        try:
            v = float(row.get(h, 0) or 0)
        except (ValueError, TypeError):
            v = 0.0
        out[canon] = v

    if out.get("duration_ms", 0) > 1e6:
        out["duration_ms"] = out["duration_ms"] / 1000.0

    out["timestamp"] = str(row.get("stime", row.get("timestamp", "")))
    out["client_ip"] = str(row.get("srcip", row.get("client_ip", "")))
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


def normalize_unsw_folder(folder: Path, output_path: Path, max_rows: int | None = None) -> int:
    """Read UNSW-NB15 CSVs, normalize to canonical schema, write one CSV. Returns row count."""
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

    LOG.info("UNSW-NB15 normalized: %d rows -> %s", written, output_path)
    return written
