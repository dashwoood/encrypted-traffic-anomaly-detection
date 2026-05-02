"""LSTM anomaly detection: recurrent classification over temporal flow windows."""
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
    fit_scaler_seq,
    scale_seq,
    load_state,
    predict_supervised,
    predict_scores_supervised,
    predict_supervised_seq,
    predict_scores_supervised_seq,
    restore_state_dict,
    save_model,
    train_supervised,
    train_supervised_seq,
)

DEFAULT_WINDOW = 16


class LSTMNet(nn.Module):
    """LSTM network for flow-sequence classification.

    Input shape: (batch, seq_len, n_features)
    The last hidden state of the top LSTM layer feeds the classifier head.
    """

    def __init__(
        self,
        n_features: int = NUM_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)  -- no reshape needed
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]           # (batch, hidden)
        return self.fc(last_hidden)     # (batch, 1)


@register
class LSTMDetector(BaseDetector):
    """LSTM: recurrent network for temporal flow-sequence analysis.

    Processes sliding windows of consecutive flows, capturing temporal
    dependencies in how flow characteristics change over time.  The window
    size controls how much temporal context each prediction receives.
    """

    name = "lstm"
    is_sequence_model = True

    def __init__(self, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
                 window_size: int = DEFAULT_WINDOW):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.window_size = window_size
        self._model: LSTMNet | None = None
        self._scaler = None

    def fit(self, flows: list[dict]) -> None:
        from datasets.sequences import flows_to_sequences
        X_seq, y = flows_to_sequences(flows, window_size=self.window_size)
        if len(X_seq) == 0:
            X_raw, y_flat = flows_to_arrays(flows)
            self._scaler, X_scaled = fit_scaler(X_raw)
            X_seq = X_scaled.reshape(len(X_scaled), 1, NUM_FEATURES)
            y = y_flat
        else:
            self._scaler, X_seq = fit_scaler_seq(X_seq)

        self._model = LSTMNet(NUM_FEATURES)
        self._model = train_supervised_seq(
            self._model, X_seq, y,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
        )

    def predict(self, flows: list[dict]) -> list[bool]:
        if self._model is None:
            self.fit(flows)
        return [s > 0.5 for s in self.predict_scores(flows)]

    def predict_scores(self, flows: list[dict]) -> list[float]:
        if self._model is None:
            self.fit(flows)
        from datasets.sequences import flows_to_sequences
        X_seq, _ = flows_to_sequences(flows, window_size=self.window_size)
        if len(X_seq) == 0:
            X_raw, _ = flows_to_arrays(flows)
            X_scaled = self._scaler.transform(X_raw).astype("float32")
            X_seq = X_scaled.reshape(len(X_scaled), 1, NUM_FEATURES)
        else:
            X_seq = scale_seq(self._scaler, X_seq)
        return predict_scores_supervised_seq(self._model, X_seq)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise ValueError("Model not fitted")
        save_model(
            path, self._model, self._scaler,
            meta={"epochs": self.epochs, "lr": self.lr, "window_size": self.window_size},
        )

    def load(self, path: Path) -> None:
        state = load_state(path)
        self._model = LSTMNet(NUM_FEATURES)
        restore_state_dict(self._model, state["state_dict"])
        self._scaler = state["scaler"]
        meta = state.get("meta", {})
        self.window_size = meta.get("window_size", self.window_size)
