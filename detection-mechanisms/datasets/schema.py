"""
Canonical flow schema for anomaly detection (reference dataset structure).

All datasets (synthetic flows.csv and benchmark CICIDS2017/UNSW-NB15) are normalized
to this schema so a single training pipeline and feature set can be used.
Structure aligns with the widely used CICIDS2017-style flow + label format.
See project docs/canonical-flow-format.md for the full column list and types.
"""
try:
    from ..flow_reader import FEATURE_COLUMNS as CANONICAL_FEATURES
    from ..flow_reader import NUM_FEATURES as CANONICAL_NUM_FEATURES
except ImportError:
    from flow_reader import FEATURE_COLUMNS as CANONICAL_FEATURES
    from flow_reader import NUM_FEATURES as CANONICAL_NUM_FEATURES

# Label column: 0 = normal, 1 = anomaly (benchmarks may use "Label"/"label"/"attack_cat")
CANONICAL_LABEL = "ground_truth"

# Full CSV header for writing normalized flows (matches receiver FLOW_HEADER)
CANONICAL_CSV_HEADER = [
    "timestamp",
    "client_ip",
    "method",
    "path",
    "path_length",
    "query_length",
    "duration_ms",
    "response_code",
    "response_size",
    "request_content_length",
    "header_count",
    "header_size_bytes",
    "user_agent",
    "user_agent_length",
    "referer",
    "referer_present",
    "accept",
    "accept_language",
    "inter_arrival_ms",
    "request_sequence",
    "requests_last_60s",
    "unique_paths_count",
    "hour_utc",
    "minute",
    "day_of_week",
    CANONICAL_LABEL,
]


def canonical_header():
    """Return the list of column names for normalized flow CSV (canonical schema)."""
    return list(CANONICAL_CSV_HEADER)
