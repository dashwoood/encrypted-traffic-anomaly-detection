# Detection Mechanisms

CLI tool and daemon for anomaly detection on flow traffic. Tracks traffic from a flows CSV and highlights anomaly vs normal using pluggable models.

## Installation

```bash
# From source
pip install -e .

# With AI models (PyTorch)
pip install -e ".[ai]"

# With benchmark dataset support (Kaggle download)
pip install -e ".[benchmark]"
# or from project root: pip install -r requirements.txt

# Build wheel
python -m build
pip install dist/*.whl
```

After installation, the `detect` command is available.

## Usage

```bash
# List available models
detect list-models

# Batch mode: process flows once
detect track --flows data/logs/flows.csv --model baseline

# Batch mode with saved model (no refit)
detect track --flows data/logs/flows.csv --model isolation_forest --model-dir data/models

# Daemon mode: watch flows file continuously (saves/loads model from --model-dir)
detect daemon --flows data/logs/flows.csv --model isolation_forest --model-dir data/models

# Train on synthetic + benchmark data and save weights (no dataset in git)
detect train --data-dir data --model isolation_forest --model-dir data/models
detect train --data-dir data --skip-download   # use existing data only (e.g. synthetic)
```

See [docs/detection-daemon.md](../docs/detection-daemon.md) for daemon setup, systemd unit, and execution details.

## Training and benchmark datasets

Training uses **benchmark datasets only** (CICIDS2017, UNSW-NB15) downloaded from Kaggle. Synthetic data is not used for evaluation. Datasets are not stored in git (`data/` is gitignored).

- **Canonical schema**: All sources are normalized to the same flow schema (17 features + `ground_truth`), aligned with CICIDS2017-style flow + label structure.
- **First-time setup (Kaggle)**  
  Install Kaggle CLI: `pip install -e ".[benchmark]"`.  
  Create `~/.kaggle/kaggle.json` with your API credentials (Kaggle → Settings → Create New Token). Or from env without storing the key in a file:
  ```bash
  export KAGGLE_USERNAME=your_username
  export KAGGLE_KEY=your_key_from_kaggle
  python scripts/setup_kaggle.py
  ```
  Then:
  ```bash
  detect train --data-dir data --model cnn
  ```
  This downloads CICIDS2017 and UNSW-NB15 (tries alternate slugs if one fails), normalizes them, writes `data/datasets/canonical_train.csv` and `data/datasets/canonical_val.csv`, fits the model, saves to `data/models/<model>.joblib`, and prints validation P/R/F1.
- **Proper evaluation (train vs test)**  
  Fit on train, report metrics on held-out test (recommended for thesis):
  ```bash
  cd experiments
  python run_experiment.py --train ../data/datasets/canonical_train.csv --test ../data/datasets/canonical_val.csv --model cnn --output results/
  python run_experiment.py --train ../data/datasets/canonical_train.csv --test ../data/datasets/canonical_val.csv --all --model-dir ../data/models
  ```

## Models

**Traditional:** `baseline`, `pca`, `kmeans`, `isolation_forest`  
**AI:** `autoencoder` (unsupervised, trained on normal flows only), `cnn`, `lstm`, `gru`, `transformer` (supervised binary classification).

Traditional models work with the base install. AI models require the `ai` extra (`pip install -e ".[ai]"` or `pip install "flow-anomaly-detector[ai]"`). Autoencoder uses reconstruction-error threshold on normal data; supervised models use weighted BCE and early stopping with a train/val split.

## Structure

```
detection-mechanisms/
├── cli.py              # CLI entry point (track, daemon, train, list-models)
├── flow_reader.py      # Read flows.csv from receiver
├── pyproject.toml      # Package config and entry point
├── datasets/           # Benchmark prep: Kaggle download, canonical normalization
│   ├── schema.py       # Canonical feature/label schema
│   ├── download.py     # Kaggle download (CICIDS2017, UNSW-NB15)
│   ├── normalize_*.py  # Map benchmark columns to canonical
│   └── prepare.py      # Merge synthetic + benchmark, train/val split
├── models/
│   ├── base.py         # BaseDetector abstract class
│   ├── registry.py     # Model registry
│   ├── traditional/    # PCA, K-means, Isolation Forest, baseline
│   └── ai/             # Autoencoder, LSTM, CNN, GRU, Transformer
└── README.md
```
