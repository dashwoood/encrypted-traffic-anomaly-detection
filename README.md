# Anomaly Detection in Encrypted Communication Using AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

Master's thesis project implementing and evaluating AI-based and traditional statistical methods for anomaly detection in encrypted network traffic.

**Author:** Bc. Aleksandra Parkhomenko
**Supervisor:** prof. RNDr. Jiří Ivánek, CSc.
**Academic Year:** 2025/2026

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Research Goals](#research-goals)
- [Implementation Components](#implementation-components)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Documentation](#documentation)

---

## Overview

This repository contains the complete implementation for a master's thesis investigating the effectiveness of artificial intelligence methods versus traditional statistical approaches for detecting anomalies in encrypted network traffic. The project addresses the challenge of identifying cyber threats in environments where the vast majority of communication is encrypted, making traditional Deep Packet Inspection (DPI) methods ineffective.

### Key Features

- **11 detection models**: Five AI (CNN, LSTM, GRU, Transformer, Autoencoder) and six traditional (Isolation Forest, PCA, K-means, Z-score threshold, Ensemble, Baseline)
- **Temporal sequence models**: LSTM, GRU, and Transformer process sliding windows of consecutive flows, capturing temporal dependencies
- **Two-level testing environment**: Synthetic HTTP mode for integration testing and benchmark replay mode for reproducible evaluation on UNSW-NB15
- **Canonical flow schema**: 17 numerical features shared across synthetic and benchmark data, enabling consistent comparison
- **Statistical hypothesis testing**: Paired Wilcoxon signed-rank test on 10-fold cross-validation, Mann-Whitney U as secondary test
- **Design Science Research**: Follows the Peffers (2007) DSR methodology

---

## Project Structure

```
diploma/
├── README.md                              # This file
├── SETUP.md                               # Docker environment setup guide
├── Makefile                               # Build, run, and evaluation targets
├── PROJECT-CZ.md                          # Project description (Czech)
├── requirements.txt                       # Python dependencies (base)
├── .env.example                           # Environment variables template
├── docker-compose.yml                     # Synthetic HTTP environment
├── docker-compose.benchmark.yml           # Benchmark replay environment
├── demo.sh                                # End-to-end demo (synthetic HTTP)
├── demo-benchmark.sh                      # End-to-end demo (benchmark replay)
├── run_experiment.sh                      # Full experiment run script
├── freeze_artifacts.sh                    # Freeze results into timestamped archive
├── ops_smoke_daemon.sh                    # Operational daemon smoke test
│
├── detection-mechanisms/                  # Detection system package
│   ├── README.md                          # Package-level documentation
│   ├── pyproject.toml                     # Package definition and dependencies
│   ├── Dockerfile                         # Container for detection daemon
│   ├── cli.py                             # CLI entry point (detect command)
│   ├── flow_reader.py                     # Canonical feature definitions + CSV reader
│   ├── evaluation.py                      # Metrics (P, R, F1, ROC-AUC, PR-AUC, FPR, FNR)
│   ├── statistical_tests.py               # Wilcoxon, Mann-Whitney U, paired t-test
│   ├── visualization.py                   # ROC/PR curves, confusion matrices, bar charts
│   ├── feature_selection.py               # Mutual information, variance threshold
│   ├── scripts/
│   │   └── setup_kaggle.py                # Kaggle API credentials helper
│   ├── datasets/
│   │   ├── schema.py                      # Canonical CSV header definition
│   │   ├── prepare.py                     # Download, normalize, split into train/val
│   │   ├── sequences.py                   # Sliding-window preparation for temporal models
│   │   ├── normalize_unsw.py              # UNSW-NB15 → canonical mapping
│   │   ├── normalize_cicids.py            # CICIDS2017 → canonical mapping
│   │   └── download.py                    # Kaggle dataset downloader
│   ├── models/
│   │   ├── base.py                        # BaseDetector abstract class
│   │   ├── registry.py                    # Model registry (get/list_models)
│   │   ├── traditional/                   # baseline, isolation_forest, pca, kmeans,
│   │   │                                  #   threshold, ensemble
│   │   └── ai/                            # autoencoder, cnn, lstm, gru, transformer
│   │       └── _common.py                 # Shared training/inference for neural models
│   └── tests/                             # Unit and integration tests
│       ├── test_cli_smoke.py
│       ├── test_registry_and_models.py
│       ├── test_evaluation_metrics.py
│       ├── test_daemon_lifecycle.py
│       ├── test_flow_replayer_behavior.py
│       └── test_experiment_outputs.py
│
├── testing-environment/                   # Containerized traffic generation
│   ├── README.md
│   ├── generator/                         # HTTP traffic generator (normal + attack patterns)
│   │   ├── Dockerfile
│   │   └── ping.py
│   ├── receiver/                          # HTTP server capturing flows to CSV
│   │   ├── Dockerfile
│   │   ├── server.py
│   │   └── entrypoint.sh
│   └── flow-replayer/                     # Replays canonical CSV rows for benchmark mode
│       ├── Dockerfile
│       └── replay_flows.py
│
├── experiments/                           # Experiment orchestration
│   ├── README.md
│   ├── run_experiment.py                  # Main experiment runner (train, evaluate, stats)
│   ├── refresh_cv_only.py                 # Re-run cross-validation without full retrain
│   └── results/                           # Output: JSON metrics, figures, run manifest
│       └── figures/                       # ROC curves, PR curves, confusion matrices
│
├── tex/                                   # LaTeX thesis source
│   ├── prace.tex                          # Main document
│   ├── literatura.bib                     # Bibliography
│   └── kap01.tex ... kap03.tex            # Chapters (Rešerše, Metodika, Výsledky)
│
├── docs/                                  # Source code documentation
│   ├── canonical-flow-format.md           # Canonical CSV schema reference
│   ├── detection-daemon.md                # Daemon mode usage
│   ├── experimental-environment.md        # Full environment setup guide
│   ├── testing-environment.md             # Docker environment architecture
│   ├── testing-environment-design.md      # Design rationale for two-mode approach
│   └── sequence-models.md                 # Temporal model windowing details
│
├── data/                                  # Data storage (gitignored)
│   ├── datasets/                          # Prepared train/val CSVs
│   │   ├── canonical_train.csv
│   │   ├── canonical_val.csv
│   │   ├── canonical_train_ordered.csv    # Temporally ordered (for sequence models)
│   │   └── canonical_val_ordered.csv
│   └── logs/                              # Runtime outputs (flows.csv, requests.log)
│
└── .github/workflows/                     # CI
    ├── package.yml                        # Python package build + test
    └── build-pdf.yml                      # LaTeX → PDF compilation
```

---

## Research Goals

### Primary Objective

Evaluate the suitability of artificial intelligence methods for anomaly detection in encrypted network communication through systematic comparison with traditional statistical methods in an experimental environment.

### Specific Objectives

1. **Literature Analysis**: Identify and analyze current AI and traditional statistical methods suitable for encrypted traffic anomaly detection through a systematic literature review with thematic analysis
2. **Environment Design**: Design a two-level experimental infrastructure (synthetic HTTP mode + benchmark replay mode) for collecting, generating, and validating network flow data
3. **Prototype Implementation**: Implement a detection system with a shared interface (`fit`, `predict`, `predict_scores`) for 11 models (5 AI, 6 traditional)
4. **Performance Evaluation**: Assess effectiveness using precision, recall, F1 score, ROC-AUC, PR-AUC, and statistical significance tests (Wilcoxon signed-rank, Mann-Whitney U)
5. **Systematic Comparison**: Compare AI and traditional method efficiency and identify situations where AI deployment is beneficial

### Hypothesis

AI methods achieve higher F1 scores than traditional statistical methods when detecting anomalies on the same dataset. Tested via paired Wilcoxon signed-rank on 10-fold cross-validation (seed 42, α = 0.05).

---

## Implementation Components

### Testing Environment (`testing-environment/`)

Two-mode containerized infrastructure:

- **Synthetic HTTP mode** (`docker-compose.yml`): Generator sends normal and attack-like HTTP requests to a receiver that captures flow metadata in [canonical format](docs/canonical-flow-format.md). Used for integration testing.
- **Benchmark replay mode** (`docker-compose.benchmark.yml`): Flow-replayer feeds pre-prepared canonical CSV rows into the same receiver pipeline. Used for reproducible evaluation on UNSW-NB15.

See [SETUP.md](SETUP.md) for Docker environment details and [docs/testing-environment.md](docs/testing-environment.md) for architecture.

### Detection Mechanisms (`detection-mechanisms/`)

#### Traditional Methods

| Model | Type | Description |
|-------|------|-------------|
| `baseline` | Baseline | Marks all flows as normal (lower bound reference) |
| `isolation_forest` | Unsupervised | Tree-based anomaly isolation |
| `pca` | Unsupervised | PCA reconstruction error threshold |
| `kmeans` | Unsupervised | Cluster distance threshold |
| `threshold` | Statistical | Z-score based per-feature thresholding |
| `ensemble` | Ensemble | Majority vote over isolation_forest, pca, kmeans |

#### AI Methods

| Model | Type | Description |
|-------|------|-------------|
| `autoencoder` | Unsupervised | Reconstruction error on normal traffic |
| `cnn` | Supervised | 1D convolution over per-flow feature vector |
| `lstm` | Supervised/Temporal | LSTM over sliding windows of 16 flows |
| `gru` | Supervised/Temporal | GRU over sliding windows of 16 flows |
| `transformer` | Supervised/Temporal | Self-attention over sliding windows of 16 flows |

Temporal models process sliding windows of consecutive flows. See [docs/sequence-models.md](docs/sequence-models.md) for windowing details.

All models implement a shared interface defined in `models/base.py`: `fit(X, y)`, `predict(X)`, `predict_scores(X)`.

---

## Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **Docker**: Version 20.10+ with Docker Compose V2 (for testing environment)
- **Hardware**: 8 GB RAM minimum; GPU optional (speeds up neural network training)

### Installation

```bash
# Clone and enter
git clone <repo-url>
cd diploma

# Configure environment
cp .env.example .env

# Install detection-mechanisms package (editable)
make install

# Or with all optional dependencies (matplotlib, Kaggle API)
make install-all
```

### Download and Prepare Datasets

Benchmark datasets (UNSW-NB15, optionally CICIDS2017) are downloaded via Kaggle API:

```bash
# Set up Kaggle credentials (~/.kaggle/kaggle.json)
cd detection-mechanisms && python scripts/setup_kaggle.py

# Prepare canonical train/val splits from UNSW-NB15
make prepare-data
```

This produces four CSV files under `data/datasets/`: shuffled train/val for per-flow models and temporally ordered train/val for sequence models.

---

## Usage

### Quick Start

```bash
# Run all 11 models with 10-fold CV and statistical tests (~23 hours on 8-core CPU)
make evaluate-all

# Results appear in experiments/results/:
#   experiment_summary.json    – per-model metrics
#   statistical_comparison.json – Wilcoxon and Mann-Whitney results
#   figures/                   – ROC, PR curves, confusion matrices
```

### CLI Commands

The `detect` CLI (installed via `make install`) provides three modes:

```bash
# List registered models
detect list-models

# Batch detection on a CSV file
detect track --flows data/logs/flows.csv --model cnn

# Daemon mode (continuous monitoring, re-reads CSV on interval)
detect daemon --flows data/logs/flows.csv --model isolation_forest --interval 5

# Train a model on benchmark data
detect train --data-dir data --model lstm --skip-download
```

### Running Experiments

```bash
# Single model
python experiments/run_experiment.py \
  --train data/datasets/canonical_train.csv \
  --test  data/datasets/canonical_val.csv \
  --model cnn --seed 42

# Sequence model (needs ordered datasets + window size)
python experiments/run_experiment.py \
  --train data/datasets/canonical_train.csv \
  --test  data/datasets/canonical_val.csv \
  --train-ordered data/datasets/canonical_train_ordered.csv \
  --test-ordered  data/datasets/canonical_val_ordered.csv \
  --model lstm --window-size 16 --seed 42

# All models with full evaluation pipeline
make evaluate-all

# Re-run cross-validation only (no full retrain)
python experiments/refresh_cv_only.py --results-dir experiments/results/ --seed 42
```

### Testing Environment

```bash
# Synthetic HTTP mode
make env-up          # start containers
make demo            # run end-to-end demo
make env-down        # stop containers

# Benchmark replay mode
make env-benchmark-up
make demo-benchmark
make env-benchmark-down

# Smoke tests
make env-smoke-http
make env-smoke-benchmark
```

### Tests

```bash
cd detection-mechanisms && python -m pytest tests/ -v
```

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision** | TP / (TP + FP) | Of predicted anomalies, how many are correct |
| **Recall** | TP / (TP + FN) | Of actual anomalies, how many are detected |
| **F1 Score** | 2 · P · R / (P + R) | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve | Threshold-independent classification quality |
| **PR-AUC** | Area under PR curve | Classification quality on imbalanced data |
| **FPR** | FP / (FP + TN) | Normal traffic incorrectly flagged |
| **FNR** | FN / (FN + TP) | Anomalies missed |

Recall is prioritized in cybersecurity contexts: missing an attack is more costly than a false alarm.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/canonical-flow-format.md](docs/canonical-flow-format.md) | Canonical CSV schema (17 features + label) |
| [docs/sequence-models.md](docs/sequence-models.md) | Sliding-window approach for LSTM/GRU/Transformer |
| [docs/detection-daemon.md](docs/detection-daemon.md) | Daemon mode usage and integration |
| [docs/experimental-environment.md](docs/experimental-environment.md) | Full environment setup and reproduction |
| [docs/testing-environment.md](docs/testing-environment.md) | Docker architecture and components |
| [docs/testing-environment-design.md](docs/testing-environment-design.md) | Design rationale for two-mode approach |
| [SETUP.md](SETUP.md) | Docker environment quick start |
