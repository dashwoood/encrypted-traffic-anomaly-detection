# Temporal Sequence Models (LSTM, GRU, Transformer)

## Motivation

Network traffic is inherently temporal -- flows arrive in chronological order and attack patterns often manifest as sequences of related events (e.g., a scanning phase followed by exploitation). Recurrent and attention-based architectures are designed to capture such temporal dependencies.

The LSTM, GRU, and Transformer models in this project process **sliding windows of consecutive flows** rather than individual flows. This gives each prediction temporal context about what happened in the recent past.

## Architecture

### Input format

```
Per-flow models (CNN, Autoencoder, traditional):
  Input:  (batch, 17)              -- one flow, 17 features

Sequence models (LSTM, GRU, Transformer):
  Input:  (batch, window_size, 17) -- window_size consecutive flows, 17 features each
```

### Sliding window construction

Given an ordered sequence of flows `[f_1, f_2, ..., f_n]` and a window size `W`:

- Window 1: `[f_1, f_2, ..., f_W]`      -> label of `f_W`
- Window 2: `[f_2, f_3, ..., f_{W+1}]`  -> label of `f_{W+1}`
- ...
- Window k: `[f_k, ..., f_{k+W-1}]`     -> label of `f_{k+W-1}`

The label of each window is the label of the **last** flow, as the model's task is to classify whether the most recent flow is anomalous given the preceding context.

### Model specifics

| Model | How it uses the sequence |
|-------|--------------------------|
| **LSTM** | Processes flows sequentially; the final hidden state captures ordered dependencies. 2 layers, 64 hidden units, dropout 0.3 |
| **GRU** | Same sequential processing with fewer parameters (2 gates vs 3). Faster training, comparable accuracy |
| **Transformer** | Self-attention over all positions in the window. Can attend to any flow regardless of distance. 2 layers, 4 heads, d_model=32 |

## Data preparation

### Ordered vs shuffled data

The project prepares two variants of the canonical dataset:

- **Shuffled** (`canonical_train.csv`, `canonical_val.csv`): Random split for per-flow models. Standard for i.i.d. classification.
- **Ordered** (`canonical_train_ordered.csv`, `canonical_val_ordered.csv`): Temporal split (first 80% train, last 20% val). Preserves row ordering for windowing. Used by sequence models.

### UNSW-NB15 temporal ordering

The UNSW-NB15 dataset preserves capture-time ordering within each CSV file. During normalization, this ordering is maintained. The "ordered" variant keeps flows in their original sequence, allowing meaningful temporal windows.

**Limitation**: The benchmark data was collected in lab conditions with specific traffic patterns. The temporal ordering reflects the capture session, not necessarily the diversity of real-world traffic sequences. This is acknowledged as a limitation.

## Usage

### From CLI

```bash
detect track --flows data/logs/flows.csv --model lstm
```

The LSTM detector internally creates sliding windows from the flow data.

### From experiments

```bash
python experiments/run_experiment.py \
  --train data/datasets/canonical_train.csv \
  --test  data/datasets/canonical_val.csv \
  --model lstm \
  --window-size 16
```

### Window size

The default window size is 16 flows. This can be adjusted:

- **Smaller windows** (8): Less context but more training samples and faster training.
- **Larger windows** (32, 64): More temporal context but fewer windows and more memory.

The window size is saved with the model and automatically restored on load.

## Comparison with per-flow approach

Per-flow models (CNN, Autoencoder, traditional methods) classify each flow independently. They rely entirely on the 17 features of a single flow. This is appropriate when:

- Flows are independent (e.g., benchmark dataset with shuffled rows)
- Anomalies are detectable from individual flow characteristics

Sequence models add value when:

- Attacks manifest as temporal patterns (e.g., periodic C2 beaconing)
- Context from neighbouring flows improves classification
- The data preserves meaningful temporal ordering
