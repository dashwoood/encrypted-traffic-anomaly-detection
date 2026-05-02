# Experimental Environment Guide

This guide explains how to run the full experimental environment for anomaly detection in encrypted traffic: from synthetic traffic generation in Docker to model evaluation and live daemon monitoring.

## Hardware and OS Requirements

- **Operating system**:
  - Linux (native) or WSL2 on Windows
  - macOS with Docker Desktop (for development only; thesis experiments were run on Linux/WSL2)
- **CPU**: 4+ cores recommended
- **RAM**: 8 GB minimum (16 GB recommended for faster training)
- **Disk space**: ≥20 GB free (datasets, Docker images, experiment outputs)
- **GPU**: Optional; the current models run on CPU. A CUDA-capable GPU can accelerate AI training but is not required.

## Software Prerequisites

- **Python**: 3.9 or newer
- **pip**: For installing Python packages
- **Docker**: Version 20.10+ with Compose V2 support
- **Make**: For convenience targets (`make ...`)
- (Optional) **Kaggle CLI**: For downloading benchmark datasets if you want to regenerate `data/datasets/`

From the project root:

```bash
make install-all
```

installs the `detection-mechanisms` package (with optional dependencies) in editable mode.

## High-Level Architecture

```text
Docker: generator  -->  Docker: receiver  -->  data/logs/flows.csv
                                    |
                                    v
                          detection mechanisms
                      (experiments + detect CLI)
```

- `testing-environment/`:
  - **generator**: produces real HTTP traffic (TCP) with a configurable anomaly rate (`ANOMALY_RATE`). When `SIMULATE_ATTACKS=1` (default), anomaly requests use simulated attack-like behaviour (suspicious User-Agents, probing paths, burst timing) so flow-level features differ from normal traffic—suitable for demos and integration tests. For real attack traces (exploits, C2, etc.) use benchmark datasets (CICIDS2017, UNSW-NB15) or external capture.
  - **receiver**: exposes an HTTP endpoint and writes each request as a flow row in `data/logs/flows.csv` (canonical schema).
- `detection-mechanisms/`:
  - **experiments**: offline evaluation via `experiments/run_experiment.py`.
  - **CLI**: `detect track` and `detect daemon` for batch and live monitoring.
- `docker-compose.yml`:
  - Orchestrates `receiver`, `generator`, and an optional `detector` service that runs the daemon inside Docker.

## Make Targets for the Environment

From the project root:

- **Environment lifecycle**
  - `make start-environment`: Start `receiver` and `generator` Docker services.
  - `make stop-environment`: Stop and remove the Docker services.
  - `make logs`: Follow combined Docker logs.
- **Evaluation**
  - `make evaluate-all`: Run all models (traditional + AI) on the canonical UNSW-NB15 train/val split.
  - `make evaluate-traditional`: Run only traditional models.
  - `make evaluate-ai`: Run only AI models (including temporal sequence models with ordered splits).
  - `make generate-thesis-results`: Run full evaluation + statistical comparison + figures.
- **Demo**
  - `./demo.sh`: One-command end-to-end demo (see below).

## Workflow 1: Quick Synthetic Demo (demo.sh)

Use this when you want to see live anomaly detection on synthetic traffic without touching datasets.

```bash
chmod +x demo.sh
./demo.sh
```

The script:

1. Ensures `data/logs/` exists.
2. Starts the Docker environment (`docker compose up -d`).
3. Waits for `data/logs/flows.csv` to appear and accumulate data.
4. Starts `detect daemon` on `data/logs/flows.csv` using the `isolation_forest` model.
5. Streams per-flow decisions to the terminal, highlighting anomalies.

Press **Ctrl+C** to stop the daemon. To stop the containers:

```bash
make stop-environment
```

## Workflow 2: Full Benchmark Experiment (UNSW-NB15)

This reproduces the thesis-level evaluation on the canonical UNSW-NB15 dataset.

```bash
make install-all
make prepare-data
make evaluate-all
make generate-thesis-results
```

- `make prepare-data`: Prepares normalized canonical train/val CSVs under `data/datasets/`.
- `make evaluate-all`: Runs all 11 models with:
  - Per-flow models on shuffled train/val splits
  - Sequence models (LSTM, GRU, Transformer) on ordered splits
  - Optional k-fold cross-validation if `--cv-folds` is passed to `run_experiment.py`.
- `make generate-thesis-results`: Produces JSON metrics, visualizations, and statistical comparison (`experiments/results/`).

## Workflow 3: Live Daemon in Docker Compose

To run the detector as a service alongside the generator and receiver:

1. Ensure `data/logs/` exists:

   ```bash
   mkdir -p data/logs
   ```

2. Start all services (including the detector):

   ```bash
   docker compose up -d
   ```

   The `detector` service:

   - Mounts `./data/logs` as `/data`.
   - Installs `detection-mechanisms` in editable mode.
   - Runs:

     ```bash
     detect daemon --flows /data/flows.csv --model isolation_forest --interval 5 --min-fit-flows 50
     ```

3. Inspect detector output via:

   ```bash
   docker compose logs -f detector
   ```

4. Stop everything when finished:

   ```bash
   make stop-environment
   ```

## Workflow 4: Benchmark Flow Replay (In-Distribution Validation)

Use this when you want to validate detectors on the **same feature space** as training. The HTTP generator produces application-level metadata; models are trained on UNSW-NB15 network-level statistics. See [testing-environment-design.md](testing-environment-design.md) for the mismatch analysis.

```bash
make prepare-data   # Ensure canonical_val.csv exists
make demo-benchmark
```

Or manually:

```bash
# 1. Ensure canonical dataset exists
make prepare-data

# 2. Start flow replayer + detector (no HTTP generator/receiver)
docker compose -f docker-compose.benchmark.yml up -d

# 3. Flow replayer streams canonical_val.csv → data/logs/flows.csv
# 4. Detector runs daemon on that file
docker compose -f docker-compose.benchmark.yml logs -f detector

# 5. Stop
docker compose -f docker-compose.benchmark.yml down
```

The flow replayer writes rows with configurable delay (`REPLAY_DELAY_MS`, default 100). Data is **in-distribution** for models trained on UNSW-NB15.

## CICIDS2017 Status

- The codebase includes normalization utilities for **CICIDS2017** in `detection-mechanisms/datasets/normalize_cicids.py`.
- CICIDS2017 can be downloaded via the Kaggle CLI and converted to the canonical schema.
- However, the **final thesis results and statistical comparisons in `docs/experiment-results.md` are based on UNSW-NB15 only**. CICIDS2017 support is provided as an extension path but is not part of the core evaluation reported in the thesis.

