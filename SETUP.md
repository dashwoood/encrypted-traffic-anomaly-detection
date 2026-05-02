# Environment Setup

## Prerequisites

- Docker 20.10+ with Docker Compose V2
- (Optional) Docker Desktop with WSL2 integration for Windows

## Quick Start

```bash
# Copy environment template (if needed)
cp .env.example .env

# Set UID/GID in .env so log files are owned by you (deletable without sudo)
# Run: id -u  and  id -g  to get values

# Create data dir (owned by you)
mkdir -p data/logs

# Build and start containers
docker compose up --build -d

# Check logs - generator should show "OK 200" (traffic reaching receiver)
docker compose logs -f generator
```

## Verify Connectivity

1. **Generator → Receiver**: The generator continuously sends HTTP requests (every 2s by default) to varied paths.
2. **Success**: `docker compose logs generator` shows `OK 200` lines.
3. **Request log**: Receiver writes every incoming request to `data/logs/requests.log`.
4. **Flow log**: `data/logs/flows.csv` – flow-level metadata (see table below).
5. **Packet capture**: `data/logs/capture.pcap` – raw packets (tcpdump) for packet-level analysis.
6. **Stop**: `docker compose down`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RECEIVER_PORT` | 8000 | Port exposed by receiver (host mapping) |
| `TRAFFIC_INTERVAL_SEC` | 2 | Seconds between simulated requests |
| `ROLLING_WINDOW_SEC` | 60 | Window for requests_last_60s rolling metric |
| `ANOMALY_RATE` | 0.1 | Fraction of requests marked as anomaly (ground truth) |

### Flow output format: canonical schema (reference dataset structure)

The receiver writes `data/logs/flows.csv` using the **canonical flow schema** so that synthetic output has the **same structure** as the reference benchmark datasets (CICIDS2017, UNSW-NB15) after normalization. One CSV = one row per flow, fixed columns, plus `ground_truth` (0=normal, 1=anomaly).

Full column list, types, and usage: **[docs/canonical-flow-format.md](docs/canonical-flow-format.md)**.

| Column | Description |
|--------|-------------|
| `timestamp` | UTC ISO8601 |
| `client_ip` | Source IP |
| `method` | HTTP method |
| `path` | URL path (no query) |
| `path_length` | Length of path |
| `query_length` | Length of query string |
| `duration_ms` | Request handling time |
| `response_code` | HTTP status |
| `response_size` | Response bytes (-1 if unknown) |
| `request_content_length` | Request body size |
| `header_count` | Number of request headers |
| `header_size_bytes` | Total size of headers |
| `user_agent` | User-Agent (truncated) |
| `user_agent_length` | Length of User-Agent |
| `referer` | Referer (truncated) |
| `referer_present` | 1 if Referer header present |
| `accept` | Accept header |
| `accept_language` | Accept-Language header |
| `inter_arrival_ms` | Time since last request from same client |
| `request_sequence` | Per-client request count |
| `requests_last_60s` | Requests from client in rolling window |
| `unique_paths_count` | Distinct paths seen from client |
| `hour_utc`, `minute`, `day_of_week` | Temporal |
| `ground_truth` | 0 = normal, 1 = anomaly (from `X-Test-Label`) |

**Run full experiment** (one command):

```bash
./run_experiment.sh
```

Or manually: `cd experiments && python3 run_experiment.py --flows ../data/logs/flows.csv --model isolation_forest`

**Live tracking** (detection-mechanisms CLI):

```bash
cd detection-mechanisms
python3 cli.py track --flows ../data/logs/flows.csv --model baseline
python3 cli.py list-models
```
