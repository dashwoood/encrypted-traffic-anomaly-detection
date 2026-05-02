# Testing Environment for Detection Mechanism Evaluation

This document describes how to use the testing environment to evaluate anomaly detection mechanisms on flow-level traffic.

## Overview

The testing environment provides:

- **Traffic generation**: Real HTTP traffic (TCP) with varied metadata (paths, headers, timing). The generator runs inside Docker and sends requests to the receiver; packets are real.
- **Anomaly / attack simulation**: A fraction of requests are marked as anomalous (ground truth via `X-Test-Label: anomaly`). By default, anomaly requests also use **simulated attack-like behaviour** so that flow metadata differs from normal traffic:
  - Suspicious User-Agents (e.g. scanner/exploit tool signatures)
  - Probing paths (e.g. `/admin`, `/.env`, path traversal style)
  - Shorter inter-request delays (bursts) so that flow-level features (e.g. `requests_last_60s`, `inter_arrival_ms`) look attack-like.
- **Flow export**: Each request is logged as a flow with full metadata and `ground_truth` (0=normal, 1=anomaly).
- **Packet capture**: Raw packets captured via tcpdump.

**What this is not**: The generator does not run real exploits or real malware. For real attack traffic (e.g. from CICIDS2017, UNSW-NB15, or red-team tools), use the benchmark dataset pipeline or external capture. This environment is for synthetic traffic with behaviourally distinct “attack” flows for integration tests and demos.

## Architecture

```
Generator (Docker)  →  Receiver (Docker)  →  flows.csv
     |                      |
     |  X-Test-Label        |  ground_truth
     |  (10% anomaly)       |  column
     v                      v
```

## How to Run an Experiment

### Option A: One command

```bash
./run_experiment.sh
```

This script: starts containers, waits 90s for flows, runs the experiment, outputs results.

### Option B: Step by step

```bash
# 1. Set UID/GID in .env (id -u / id -g) so log files are deletable by you
# 2. Create data dir
mkdir -p data/logs

# 3. Start environment
docker compose up -d

# 4. Wait for flows (90s recommended)
sleep 90

# 5. Run experiment (all traditional models)
cd experiments
python3 run_experiment.py --flows ../data/logs/flows.csv --all

# 6. Check results
cat results/experiment_isolation_forest.json
```

### Example output

```json
{
  "precision": 0.25,
  "recall": 0.5,
  "f1": 0.3333,
  "tp": 2, "fp": 6, "fn": 2, "tn": 69,
  "n_flows": 79,
  "n_anomalies_true": 4,
  "n_anomalies_pred": 8,
  "model": "isolation_forest",
  "flows_file": "../data/logs/flows.csv"
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANOMALY_RATE` | 0.1 | Fraction of generator requests marked as anomaly (ground truth) |
| `SIMULATE_ATTACKS` | 1 | If 1/true/yes, anomaly requests use attack-like paths, User-Agents, and burst timing so flow metadata differs from normal traffic |
| `TRAFFIC_INTERVAL_SEC` | 2 | Seconds between normal requests |
| `ROLLING_WINDOW_SEC` | 60 | Window for `requests_last_60s` metric |

## Output format: canonical flow schema

The receiver captures each request as one row in `flows.csv` using the **canonical flow schema** (reference dataset structure). This is the **same or similar structure** as the normalized benchmark datasets (CICIDS2017, UNSW-NB15), so synthetic and benchmark data can be merged for training. Column order and names are fixed; see **[canonical-flow-format.md](canonical-flow-format.md)** for the full list and types.

## Output Files

| File | Description |
|------|-------------|
| `data/logs/flows.csv` | Flow-level metadata (canonical schema), one row per request, `ground_truth` column |
| `data/logs/requests.log` | Human-readable request log |
| `data/logs/capture.pcap` | Raw packet capture |
| `experiments/results/experiment_<model>.json` | Evaluation metrics (precision, recall, F1) |

## Evaluation Metrics

The experiment script computes:

- **Precision**: Of predicted anomalies, how many are correct
- **Recall**: Of actual anomalies, how many were detected
- **F1**: Harmonic mean of precision and recall
- **Confusion matrix**: TP, FP, FN, TN

## Supported Models

- **Traditional**: `baseline`, `isolation_forest`, `pca`, `kmeans`, `threshold`, `ensemble`
- **AI**: `autoencoder`, `cnn`, `lstm`, `gru`, `transformer`

See [experiment-results.md](experiment-results.md) for evaluation results.

## Two Modes: HTTP Synthetic vs Benchmark Replay

The HTTP generator above produces **application-level** flow metadata (paths, headers). Models are trained on **benchmark datasets** (UNSW-NB15, CICIDS2017), which use **network-level** statistics (packet sizes, TTLs, inter-packet times). The feature distributions differ—detectors trained on benchmarks may perform poorly on HTTP synthetic flows. See [testing-environment-design.md](testing-environment-design.md) for details.

For validating detectors on **in-distribution** data (same as training), use the **benchmark flow replayer**:

```bash
make demo-benchmark
# Or: docker compose -f docker-compose.benchmark.yml up -d
```

The flow replayer streams rows from `data/datasets/canonical_val.csv` into `flows.csv`, so the detector sees the exact feature space it was trained on.

## Notes

- Clear `data/logs/flows.csv` and restart containers to get fresh flows with the full schema and `ground_truth` column after updating the receiver.
- Results depend on traffic patterns and ANOMALY_RATE; run longer for more stable metrics.
