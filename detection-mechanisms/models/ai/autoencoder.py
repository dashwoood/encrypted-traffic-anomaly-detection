"""Autoencoder anomaly detection: reconstruction error as anomaly score."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseDetector
from ..registry import register
from ._common import (
    NUM_FEATURES,
    SEED,
    flows_to_arrays,
    fit_scaler,
    get_device,
    load_state,
    restore_state_dict,
    save_model,
)

LOG = logging.getLogger(__name__)


class AutoencoderNet(nn.Module):
    def __init__(self, n_features: int = NUM_FEATURES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@register
class AutoencoderDetector(BaseDetector):
    """Autoencoder: unsupervised anomaly detection via reconstruction error.

    Trains to reconstruct scaled flow features; flows with high reconstruction
    error (above a contamination-based percentile threshold) are flagged as anomalies.
    """

    name = "autoencoder"

    def __init__(
        self,
        contamination: float = 0.1,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 15,
    ):
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self._model: AutoencoderNet | None = None
        self._scaler = None
        self._threshold: float = 0.0

    def fit(self, flows: list[dict]) -> None:
        """Train on normal flows only (unsupervised). Threshold from normal reconstruction errors."""
        torch.manual_seed(SEED)
        X_raw, y = flows_to_arrays(flows)
        normal_mask = (y == 0)
        n_normal = int(normal_mask.sum())
        if n_normal == 0:
            LOG.warning("No normal flows in training data; using all flows (unsupervised setup broken)")
            X_train_raw = X_raw
        else:
            X_train_raw = X_raw[normal_mask]
        self._scaler, X_train = fit_scaler(X_train_raw)

        device = get_device()
        self._model = AutoencoderNet(NUM_FEATURES).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        dataset = TensorDataset(torch.from_numpy(X_train))
        bs = min(self.batch_size, len(X_train))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        best_loss = float("inf")
        best_state: dict | None = None
        wait = 0

        for epoch in range(self.epochs):
            self._model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = self._model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(X_train)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    LOG.info("Autoencoder early stop at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.cpu().eval()

        # Threshold from reconstruction errors on normal training data only
        errors_normal = self._reconstruction_errors(X_train)
        self._threshold = float(np.percentile(errors_normal, 100 * (1 - self.contamination)))
        LOG.info(
            "Autoencoder fitted on %d normal flows; threshold = %.4f (contamination=%.2f)",
            len(X_train), self._threshold, self.contamination,
        )

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X)
            recon = self._model(X_t).numpy()
        return np.sum((X - recon) ** 2, axis=1)

    def predict(self, flows: list[dict]) -> list[bool]:
        if self._model is None:
            self.fit(flows)
        X_raw, _ = flows_to_arrays(flows)
        X = self._scaler.transform(X_raw).astype(np.float32)
        errors = self._reconstruction_errors(X)
        return [bool(e > self._threshold) for e in errors]

    def predict_scores(self, flows: list[dict]) -> list[float]:
        """Return normalised reconstruction error as anomaly score (0..1)."""
        if self._model is None:
            self.fit(flows)
        X_raw, _ = flows_to_arrays(flows)
        X = self._scaler.transform(X_raw).astype(np.float32)
        errors = self._reconstruction_errors(X)
        if self._threshold > 0:
            scores = errors / (2.0 * self._threshold)
        else:
            scores = errors
        return [min(float(s), 1.0) for s in scores]

    def save(self, path: Path) -> None:
        if self._model is None:
            raise ValueError("Model not fitted")
        save_model(
            path, self._model, self._scaler,
            threshold=self._threshold,
            meta={
                "contamination": self.contamination,
                "epochs": self.epochs,
                "lr": self.lr,
            },
        )

    def load(self, path: Path) -> None:
        state = load_state(path)
        self._model = AutoencoderNet(NUM_FEATURES)
        restore_state_dict(self._model, state["state_dict"])
        self._scaler = state["scaler"]
        self._threshold = state["threshold"]
        meta = state.get("meta", {})
        self.contamination = meta.get("contamination", self.contamination)
