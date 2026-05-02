"""Feature selection utilities for flow anomaly detection.

Provides mutual-information and variance-threshold methods for identifying
the most informative features from the canonical 17-feature set.
"""
import logging
from typing import Literal

import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from flow_reader import FEATURE_COLUMNS

LOG = logging.getLogger(__name__)


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: Literal["mutual_info", "variance"] = "mutual_info",
    top_k: int = 15,
) -> tuple[list[int], list[str]]:
    """Select the *top_k* most informative features.

    Parameters
    ----------
    X : ndarray, shape (n, 17)
    y : ndarray, shape (n,)  -- ignored for variance method
    method : "mutual_info" or "variance"
    top_k : number of features to keep

    Returns
    -------
    indices : list of column indices (into FEATURE_COLUMNS)
    names   : corresponding feature names
    """
    top_k = min(top_k, X.shape[1])

    if method == "mutual_info":
        scores = mutual_info_classif(X, y, random_state=42)
        order = np.argsort(scores)[::-1]
        indices = order[:top_k].tolist()
    elif method == "variance":
        vt = VarianceThreshold()
        vt.fit(X)
        variances = vt.variances_
        order = np.argsort(variances)[::-1]
        indices = order[:top_k].tolist()
    else:
        raise ValueError(f"Unknown method: {method}")

    names = [FEATURE_COLUMNS[i] for i in indices]
    LOG.info("Selected %d features (%s): %s", top_k, method, names)
    return indices, names


def apply_selection(X: np.ndarray, indices: list[int]) -> np.ndarray:
    """Return X with only the selected column indices."""
    return X[:, indices]
