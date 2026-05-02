#!/bin/bash
# Run detection experiments on benchmark data (CICIDS2017 + UNSW-NB15). No synthetic data.
set -e

cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-data}"

echo "1. Preparing benchmark datasets (download from Kaggle if needed)..."
python3 -c "
import sys
sys.path.insert(0, 'detection-mechanisms')
from datasets.prepare import prepare_datasets
prepare_datasets('$DATA_DIR', download_benchmarks=True)
"
# Alternative: detect train --data-dir "$DATA_DIR" --model isolation_forest  # fits one model; we use prepare only

echo "2. Running experiments (train on canonical_train, evaluate on canonical_val)..."
cd experiments
python3 run_experiment.py \
  --train "../$DATA_DIR/datasets/canonical_train.csv" \
  --test "../$DATA_DIR/datasets/canonical_val.csv" \
  --train-ordered "../$DATA_DIR/datasets/canonical_train_ordered.csv" \
  --test-ordered "../$DATA_DIR/datasets/canonical_val_ordered.csv" \
  --all \
  --model-dir "../$DATA_DIR/models" \
  --output results

echo "3. Done. Results in experiments/results/"
