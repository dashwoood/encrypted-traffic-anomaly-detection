"""Shared utilities for PyTorch-based anomaly detection models."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

_DETECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DETECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DETECT_ROOT))

from flow_reader import FEATURE_COLUMNS, NUM_FEATURES, flows_to_arrays  # noqa: E402

LOG = logging.getLogger(__name__)

SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


try:
    from imblearn.over_sampling import SMOTE as _SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False


def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class with SMOTE if imbalanced-learn is installed."""
    if not _HAS_SMOTE:
        LOG.warning("imbalanced-learn not installed; skipping SMOTE")
        return X, y
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ratio = min(n_pos, n_neg) / max(n_pos, n_neg, 1)
    if ratio > 0.33:
        return X, y
    sm = _SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X, y)
    LOG.info("SMOTE: %d -> %d samples", len(X), len(X_res))
    return X_res.astype(np.float32), y_res.astype(np.float32)


class FlowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def fit_scaler(X: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    return scaler, X_scaled


def train_supervised(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    val_split: float = 0.2,
    recall_priority: bool = True,
    use_smote: bool = False,
) -> nn.Module:
    """Train a binary classifier with BCE loss, class weighting, and early stopping.

    When *recall_priority* is True (default) the positive class (anomaly) is
    never down-weighted, ensuring the model favours detection over precision.
    This matters when anomalies are the majority class in the dataset.
    """
    torch.manual_seed(SEED)
    device = get_device()
    model = model.to(device)

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos == 0 or n_neg == 0:
        LOG.warning("Only one class in training data (%d pos, %d neg); skipping training", n_pos, n_neg)
        model.eval()
        return model

    pw = n_neg / n_pos
    if recall_priority:
        pw = max(pw, 1.0)
    pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = FlowDataset(X_train, y_train)
    n_val = max(1, int(len(dataset) * val_split))
    n_train_split = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train_split, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    bs = min(batch_size, max(1, n_train_split))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)

    best_val_loss = float("inf")
    best_state: dict | None = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b).squeeze(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b).squeeze(-1)
                val_loss += criterion(logits, y_b).item() * len(X_b)
        val_loss /= max(len(val_ds), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                LOG.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return model


def predict_supervised(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> list[bool]:
    """Run inference, return anomaly flags."""
    scores = predict_scores_supervised(model, X, batch_size=batch_size)
    return [s > 0.5 for s in scores]


def predict_scores_supervised(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> list[float]:
    """Run inference, return anomaly probabilities (0..1)."""
    device = get_device()
    model = model.to(device).eval()
    X_t = torch.from_numpy(X).to(device)
    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i : i + batch_size]
            logits = model(batch).squeeze(-1)
            probs = torch.sigmoid(logits)
            scores.extend(probs.cpu().tolist())
    return scores


class SequenceFlowDataset(Dataset):
    """Dataset of (window, label) pairs for temporal models."""

    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X_seq)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def fit_scaler_seq(X_seq: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """Fit a scaler on sequence data of shape (n, window, features).

    The scaler is fit on the flattened (n*window, features) view so that
    every time-step uses the same scaling parameters.
    """
    n, w, f = X_seq.shape
    flat = X_seq.reshape(-1, f)
    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat).astype(np.float32)
    return scaler, flat_scaled.reshape(n, w, f)


def scale_seq(scaler: StandardScaler, X_seq: np.ndarray) -> np.ndarray:
    n, w, f = X_seq.shape
    flat = X_seq.reshape(-1, f)
    return scaler.transform(flat).astype(np.float32).reshape(n, w, f)


def train_supervised_seq(
    model: nn.Module,
    X_seq: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    val_split: float = 0.2,
    recall_priority: bool = True,
) -> nn.Module:
    """Train a sequence-based binary classifier (same logic as train_supervised)."""
    torch.manual_seed(SEED)
    device = get_device()
    model = model.to(device)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        LOG.warning("Only one class (%d pos, %d neg); skipping training", n_pos, n_neg)
        model.eval()
        return model

    pw = n_neg / n_pos
    if recall_priority:
        pw = max(pw, 1.0)
    pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = SequenceFlowDataset(X_seq, y)
    n_val = max(1, int(len(dataset) * val_split))
    n_train_split = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train_split, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    bs = min(batch_size, max(1, n_train_split))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)

    best_val_loss = float("inf")
    best_state: dict | None = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b).squeeze(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b).squeeze(-1)
                val_loss += criterion(logits, y_b).item() * len(X_b)
        val_loss /= max(len(val_ds), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                LOG.info("Seq early stop at epoch %d (patience=%d)", epoch + 1, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return model


def predict_scores_supervised_seq(model: nn.Module, X_seq: np.ndarray, batch_size: int = 512) -> list[float]:
    """Inference on sequence data, return anomaly probabilities."""
    device = get_device()
    model = model.to(device).eval()
    X_t = torch.from_numpy(X_seq).to(device)
    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i: i + batch_size]
            logits = model(batch).squeeze(-1)
            probs = torch.sigmoid(logits)
            scores.extend(probs.cpu().tolist())
    return scores


def predict_supervised_seq(model: nn.Module, X_seq: np.ndarray, batch_size: int = 512) -> list[bool]:
    """Inference on sequence data, return anomaly flags."""
    return [s > 0.5 for s in predict_scores_supervised_seq(model, X_seq, batch_size=batch_size)]


def save_model(path: Path, model: nn.Module, scaler: StandardScaler, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "scaler": scaler,
        **extra,
    }
    joblib.dump(state, path)


def load_state(path: Path) -> dict:
    return joblib.load(path)


def restore_state_dict(model: nn.Module, raw: dict) -> nn.Module:
    sd = {k: torch.from_numpy(v) for k, v in raw.items()}
    model.load_state_dict(sd)
    model.eval()
    return model
