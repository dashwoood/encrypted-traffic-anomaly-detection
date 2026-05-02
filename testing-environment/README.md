# Testing Environment

Traffic generation and capture infrastructure for the anomaly detection experiments.

## Components

- **Generator** (`generator/ping.py`): Sends real HTTP requests to the receiver at a configurable interval. A fraction of requests are marked as anomalous (`X-Test-Label: anomaly`). With `SIMULATE_ATTACKS=1` (default), anomaly requests use attack-like behaviour (scanner User-Agents, probing paths, burst timing) so flow metadata differs from normal traffic.
- **Receiver** (`receiver/server.py`): HTTP server that logs each request as a **flow row** in `flows.csv`.

## Output format: same structure as reference dataset

The receiver writes **flows.csv** using the **canonical flow schema** so that the synthetic output has the **same structure** as the reference benchmark datasets (CICIDS2017, UNSW-NB15) after normalization. This allows:

- Training on both synthetic and benchmark data without schema conversion.
- Consistent column order and types for all flow CSVs in the project.

Full column list and rules: **[../docs/canonical-flow-format.md](../docs/canonical-flow-format.md)**.

## Running

From project root:

```bash
mkdir -p data/logs
docker compose up -d
# Wait for traffic (e.g. 90s), then use data/logs/flows.csv for experiments or detect train
```

See [SETUP.md](../SETUP.md) and [docs/testing-environment.md](../docs/testing-environment.md) for details.
