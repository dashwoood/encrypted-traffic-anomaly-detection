"""Transformer anomaly detection: self-attention over temporal flow windows."""
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
    predict_scores_supervised_seq,
    restore_state_dict,
    save_model,
    train_supervised_seq,
)

DEFAULT_WINDOW = 16


class TransformerNet(nn.Module):
    """Transformer encoder for flow-sequence binary classification.

    Each time-step's 17 features are projected to d_model dimensions, augmented
    with learnable positional embeddings, and passed through a stack of
    transformer encoder layers.  Global average pooling over the sequence
    dimension produces a fixed-size representation for the classification head.
    """

    def __init__(
        self,
        n_features: int = NUM_FEATURES,
        window_size: int = DEFAULT_WINDOW,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)                            # (batch, seq, d_model)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]        # add positional info
        x = self.encoder(x)                                # (batch, seq, d_model)
        x = x.mean(dim=1)                                  # global avg pool
        return self.fc(x)                                   # (batch, 1)


@register
class TransformerDetector(BaseDetector):
    """Transformer: attention-based architecture for temporal flow-sequence modeling.

    Self-attention allows the model to weigh relationships between any pair of
    time-steps regardless of distance, capturing complex temporal interactions
    that convolutional or recurrent models may miss.
    """

    name = "transformer"
    is_sequence_model = True

    def __init__(self, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
                 window_size: int = DEFAULT_WINDOW):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.window_size = window_size
        self._model: TransformerNet | None = None
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

        self._model = TransformerNet(NUM_FEATURES, window_size=X_seq.shape[1])
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
        meta = state.get("meta", {})
        ws = meta.get("window_size", self.window_size)
        self._model = TransformerNet(NUM_FEATURES, window_size=ws)
        restore_state_dict(self._model, state["state_dict"])
        self._scaler = state["scaler"]
        self.window_size = ws
