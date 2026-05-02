# Canonical flow output format (reference dataset structure)

All flow data used for training and evaluation in this project uses a **single canonical schema**. The synthetic output from the testing environment and the normalized benchmark datasets (CICIDS2017, UNSW-NB15) share this structure so the same models and pipelines can be used on any source.

The format is aligned with common intrusion-detection flow datasets (CICIDS2017-style): one row per flow, fixed numeric features plus a label column.

## CSV structure

- **File**: CSV with header row, UTF-8 encoding.
- **Header**: Exactly the column names below, in this order.
- **Label column**: `ground_truth` — `0` = normal, `1` = anomaly.

### Column list (order matters)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | UTC time (e.g. ISO8601) |
| `client_ip` | string | Source IP |
| `method` | string | HTTP method (e.g. GET, POST) |
| `path` | string | URL path (no query) |
| `path_length` | number | Length of path |
| `query_length` | number | Length of query string |
| `duration_ms` | number | Request/flow duration (milliseconds) |
| `response_code` | number | HTTP status or protocol code |
| `response_size` | number | Response size (bytes; -1 if unknown) |
| `request_content_length` | number | Request body size (bytes) |
| `header_count` | number | Number of headers |
| `header_size_bytes` | number | Total header size (bytes) |
| `user_agent` | string | User-Agent (truncated) |
| `user_agent_length` | number | User-Agent length |
| `referer` | string | Referer (truncated) |
| `referer_present` | number | 1 if Referer present, else 0 |
| `accept` | string | Accept header |
| `accept_language` | string | Accept-Language header |
| `inter_arrival_ms` | number | Time since previous flow from same client (ms); empty or -1 if first |
| `request_sequence` | number | Per-client request count |
| `requests_last_60s` | number | Requests from client in last 60s (configurable window) |
| `unique_paths_count` | number | Distinct paths seen from client |
| `hour_utc` | number | Hour (0–23) |
| `minute` | number | Minute (0–59) |
| `day_of_week` | number | Weekday (0–6) |
| `ground_truth` | number | **0** = normal, **1** = anomaly |

## Feature set used by models

Detection models use only the following **17 numeric columns** (and ignore the rest for training):

`path_length`, `query_length`, `duration_ms`, `response_code`, `response_size`, `request_content_length`, `header_count`, `header_size_bytes`, `user_agent_length`, `referer_present`, `inter_arrival_ms`, `request_sequence`, `requests_last_60s`, `unique_paths_count`, `hour_utc`, `minute`, `day_of_week`.

The same feature set is defined in `detection-mechanisms/flow_reader.py` (`FEATURE_COLUMNS`) and used when normalizing benchmark datasets.

## UNSW-NB15 mapping quality

The canonical schema was originally designed for HTTP flow metadata. When normalizing UNSW-NB15 data (low-level network statistics), several fields are **approximate mappings** rather than direct equivalents:

| Canonical field | UNSW-NB15 source | Quality |
|----------------|-------------------|---------|
| `path_length` | `smean` (mean source packet size) | Approximate: both are size-like, but semantically different |
| `query_length` | `dmean` (mean dest packet size) | Approximate |
| `duration_ms` | `dur` (flow duration in seconds) | Unit mismatch: seconds vs milliseconds (auto-converted) |
| `response_size` | `dbytes` (dest-to-source bytes) | Good match |
| `request_content_length` | `sbytes` (source-to-dest bytes) | Good match |
| `header_count` | `sttl` (source TTL, 0-255) | Approximate: both are small integers describing connection metadata |
| `header_size_bytes` | `dttl` (dest TTL) | Approximate |
| `user_agent_length` | - | Not available (set to 0) |
| `referer_present` | - | Not available (set to 0) |
| `inter_arrival_ms` | `sinpkt` (source inter-packet arrival) | Reasonable match |
| `request_sequence` | `spkts` (source packet count) | Reasonable match |
| `requests_last_60s` | `dpkts` (dest packet count) | Approximate |
| `unique_paths_count` | `ct_srv_src` (connections to same service) | Approximate |
| `hour_utc` | `ct_src_ltm` (connections from src in last 100 records) | Approximate: temporal activity proxy |
| `minute` | `ct_dst_ltm` (connections to dest in last 100 records) | Approximate: temporal activity proxy |
| `day_of_week` | - | Not available (set to 0) |

**Impact**: The approximate mappings mean that absolute feature values differ between synthetic and UNSW-NB15 data. However, since all models are trained and evaluated on the *same* normalized representation, the relative comparison between methods remains valid. This limitation is documented in the thesis methodology.

## Where this format is used

| Source | File(s) | Notes |
|--------|---------|--------|
| **Synthetic (testing env)** | `data/logs/flows.csv` | Receiver writes one row per HTTP request; header and column order match this document. |
| **CICIDS2017** | Normalized to canonical | `detection-mechanisms/datasets/normalize_cicids.py` maps benchmark columns → canonical. |
| **UNSW-NB15** | Normalized to canonical | `detection-mechanisms/datasets/normalize_unsw.py` maps benchmark columns → canonical. |
| **Prepared train/val (shuffled)** | `canonical_train.csv`, `canonical_val.csv` | Random split for per-flow models. |
| **Prepared train/val (ordered)** | `canonical_train_ordered.csv`, `canonical_val_ordered.csv` | Temporal split (first 80% / last 20%) for sequence models. |

## Consistency rules for writers

- Use **empty string** for missing numeric values when appropriate (readers coerce to 0).
- Use **UTF-8** for the CSV file.
- Use **comma** as delimiter; quote fields only when necessary (e.g. if value contains comma).
- `ground_truth` must be present and 0 or 1 for supervised evaluation.
