# Experiments

## Run full experiment (recommended)

From project root:

```bash
./run_experiment.sh
```

This prepares benchmark data (CICIDS2017 + UNSW-NB15 from Kaggle), runs all models on train/val split, and saves results to `results/`. Requires Kaggle credentials (~/.kaggle/kaggle.json).

## Manual run

```bash
# 1. Prepare benchmark data (download from Kaggle if needed)
cd .. && detect train --data-dir data --model isolation_forest
# Or: python3 -c "import sys; sys.path.insert(0,'detection-mechanisms'); from datasets.prepare import prepare_datasets; prepare_datasets('data')"

# 2. Run experiments (train on canonical_train, evaluate on canonical_val)
cd experiments
python3 run_experiment.py --train ../data/datasets/canonical_train.csv --test ../data/datasets/canonical_val.csv --all

# Single model
python3 run_experiment.py --train ../data/datasets/canonical_train.csv --test ../data/datasets/canonical_val.csv --model cnn
```

## Options

| Option | Description |
|--------|-------------|
| `--train` | Path to training flows CSV (required with --test) |
| `--test` | Path to test flows CSV for evaluation |
| `--flows` | Legacy: fit and eval on same file (not recommended) |
| `--model` | Model name (default: isolation_forest) |
| `--all` | Run all models (traditional + AI) |
| `--output` | Output directory (default: results/) |
| `--model-dir` | Load/save model weights; skip refit if file exists |

## Output

- `results/experiment_<model>.json` — precision, recall, F1 per model
- `results/experiment_summary.json` — combined results (when using `--all`)

See [docs/experiment-results.md](../docs/experiment-results.md) for methodology.
