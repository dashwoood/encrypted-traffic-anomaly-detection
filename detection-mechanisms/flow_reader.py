"""Read flow data from CSV (flows.csv format from receiver).

This module defines the canonical feature list (FEATURE_COLUMNS) used by all
detection models. Other modules should import from here rather than
re-declaring the list.
"""
import csv
from pathlib import Path
from typing import Iterator

import numpy as np

FEATURE_COLUMNS = [
    "path_length", "query_length", "duration_ms", "response_code", "response_size",
    "request_content_length", "header_count", "header_size_bytes", "user_agent_length",
    "referer_present", "inter_arrival_ms", "request_sequence", "requests_last_60s",
    "unique_paths_count", "hour_utc", "minute", "day_of_week",
]

NUM_FEATURES = len(FEATURE_COLUMNS)

_NUMERIC_FIELDS = FEATURE_COLUMNS + ["ground_truth"]

# Header cache for incremental reads (keyed by file path)
_header_cache: dict[str, list[str]] = {}


def read_flows(path: Path) -> Iterator[dict]:
    """Yield flow rows as dicts from CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield _coerce_row(row)


def _coerce_row(row: dict) -> dict:
    """Convert string values to appropriate types for model input."""
    out = dict(row)
    for k in _NUMERIC_FIELDS:
        if k in out:
            try:
                out[k] = float(out[k]) if out[k] else 0.0
            except (ValueError, TypeError):
                out[k] = 0.0
    return out


def flow_to_features(flow: dict) -> list[float]:
    """Extract numeric feature vector for model input. Order matches FEATURE_COLUMNS."""
    return [float(flow.get(c, 0) or 0) for c in FEATURE_COLUMNS]


def flows_to_feature_array(flows: list[dict], dtype=np.float64) -> np.ndarray:
    """Convert flow dicts to a (n, NUM_FEATURES) numpy array."""
    return np.array([flow_to_features(f) for f in flows], dtype=dtype)


def flows_to_arrays(flows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) from flow dicts.  X: (n, 17) float32, y: (n,) float32."""
    X = flows_to_feature_array(flows, dtype=np.float32)
    y = np.array(
        [int(float(f.get("ground_truth", 0) or 0)) for f in flows],
        dtype=np.float32,
    )
    return X, y


def read_flows_incremental(path: Path, start_byte: int = 0) -> tuple[list[dict], int]:
    """Read flows from path starting at start_byte. Returns (flows, end_byte) for daemon tailing."""
    path_key = str(path)
    flows: list[dict] = []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            if start_byte == 0:
                reader = csv.DictReader(f)
                _header_cache[path_key] = reader.fieldnames or []
            else:
                cached = _header_cache.get(path_key)
                if cached is None:
                    header_line = f.readline()
                    cached = [h.strip() for h in header_line.split(",")]
                    _header_cache[path_key] = cached
                f.seek(start_byte)
                reader = csv.DictReader(f, fieldnames=cached)
            for row in reader:
                flows.append(_coerce_row(row))
            end_byte = f.tell()
    except (FileNotFoundError, PermissionError):
        return [], start_byte
    return flows, end_byte
