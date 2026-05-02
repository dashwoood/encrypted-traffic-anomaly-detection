"""1D-CNN anomaly detection: convolutional feature extraction + binary classification."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..base import BaseDetector
from ..registry import register
from ._common import (
    NUM_FEATURES,
    flows_to_arrays,
    fit_scaler,
    load_state,
    predict_supervised,
    predict_scores_supervised,
    restore_state_dict,
    save_model,
    train_supervised,
)


class CNNNet(nn.Module):
    """1D convolutional network for flow-level binary classification.

    Input shape: (batch, NUM_FEATURES).
    Internally reshaped to (batch, 1, NUM_FEATURES) for Conv1d.
    """

    def __init__(self, n_features: int = NUM_FEATURES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (batch, 1, features)
        x = self.conv(x).squeeze(2)  # (batch, 64)
        return self.fc(x)            # (batch, 1)


@register
class CNNDetector(BaseDetector):
    """1D-CNN: supervised anomaly detection using convolutional feature extraction.

    Two Conv1d layers extract local patterns from the 17-dimensional flow feature
    vector, followed by global average pooling and fully connected classification.
    Trained with weighted BCE loss to handle class imbalance.
    """

    name = "cnn"

    def __init__(self, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model: CNNNet | None = None
        self._scaler = None

    def fit(self, flows: list[dict]) -> None:
        X_raw, y = flows_to_arrays(flows)
        self._scaler, X = fit_scaler(X_raw)
        self._model = CNNNet(NUM_FEATURES)
        self._model = train_supervised(
            self._model, X, y,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
        )

    def predict(self, flows: list[dict]) -> list[bool]:
        if self._model is None:
            self.fit(flows)
        X_raw, _ = flows_to_arrays(flows)
        X = self._scaler.transform(X_raw).astype("float32")
        return predict_supervised(self._model, X)

    def predict_scores(self, flows: list[dict]) -> list[float]:
        if self._model is None:
            self.fit(flows)
        X_raw, _ = flows_to_arrays(flows)
        X = self._scaler.transform(X_raw).astype("float32")
        return predict_scores_supervised(self._model, X)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise ValueError("Model not fitted")
        save_model(
            path, self._model, self._scaler,
            meta={"epochs": self.epochs, "lr": self.lr},
        )

    def load(self, path: Path) -> None:
        state = load_state(path)
        self._model = CNNNet(NUM_FEATURES)
        restore_state_dict(self._model, state["state_dict"])
        self._scaler = state["scaler"]
