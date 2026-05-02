"""
Sliding-window sequence preparation for temporal models (LSTM, GRU, Transformer).

Reads a canonical CSV in row order, extracts feature windows of
*window_size* consecutive flows, and assigns the label of the **last** flow
in each window as the target.

Row order is treated as implicit temporal ordering.  For benchmark datasets
(UNSW-NB15) this reflects the capture-time ordering preserved during
normalisation; the ``prepare_datasets_ordered`` helper in ``prepare.py``
produces un-shuffled train/val splits suitable for windowing.
"""
import csv
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

try:
    from ..flow_reader import FEATURE_COLUMNS, NUM_FEATURES, flow_to_features
except ImportError:
    import sys
    _DETECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if _DETECT_ROOT not in sys.path:
        sys.path.insert(0, _DETECT_ROOT)
    from flow_reader import FEATURE_COLUMNS, NUM_FEATURES, flow_to_features


def prepare_sequence_data(
    flows_csv: Path,
    window_size: int = 16,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from a canonical CSV.

    Returns
    -------
    X_seq : ndarray, shape (n_windows, window_size, NUM_FEATURES)
    y     : ndarray, shape (n_windows,)   label of last flow in each window
    """
    features: list[list[float]] = []
    labels: list[int] = []

    with open(flows_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            features.append(flow_to_features(row))
            gt = row.get("ground_truth", 0)
            try:
                labels.append(1 if int(float(gt or 0)) else 0)
            except (ValueError, TypeError):
                labels.append(0)

    n = len(features)
    if n < window_size:
        LOG.warning("Not enough rows (%d) for window_size=%d", n, window_size)
        return np.empty((0, window_size, NUM_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.float32)

    feat_arr = np.array(features, dtype=np.float32)
    label_arr = np.array(labels, dtype=np.float32)

    indices = range(0, n - window_size + 1, stride)
    X_seq = np.stack([feat_arr[i:i + window_size] for i in indices])
    y = label_arr[[i + window_size - 1 for i in indices]]

    LOG.info(
        "Prepared %d windows (window_size=%d, stride=%d) from %d flows",
        len(X_seq), window_size, stride, n,
    )
    return X_seq, y


def flows_to_sequences(
    flows: list[dict],
    window_size: int = 16,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create windows from an in-memory list of flow dicts.

    Same semantics as ``prepare_sequence_data`` but works on already-loaded
    data (e.g. inside a detector's ``fit`` / ``predict``).
    """
    feat = np.array([flow_to_features(f) for f in flows], dtype=np.float32)
    labels = np.array(
        [int(float(f.get("ground_truth", 0) or 0)) for f in flows],
        dtype=np.float32,
    )

    n = len(feat)
    if n < window_size:
        return np.empty((0, window_size, NUM_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.float32)

    indices = range(0, n - window_size + 1, stride)
    X_seq = np.stack([feat[i:i + window_size] for i in indices])
    y = labels[[i + window_size - 1 for i in indices]]
    return X_seq, y
