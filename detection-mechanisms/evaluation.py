"""Comprehensive evaluation metrics for anomaly detection models.

Replaces the hand-rolled compute_metrics in run_experiment.py with a full
metric suite including ROC-AUC, PR-AUC, FPR, and FNR.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_full_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[bool] | np.ndarray,
    y_scores: list[float] | np.ndarray | None = None,
) -> dict:
    """Compute a comprehensive set of evaluation metrics.

    Parameters
    ----------
    y_true   : ground-truth labels (0 = normal, 1 = anomaly)
    y_pred   : binary predictions (True/1 = anomaly)
    y_scores : continuous anomaly scores (higher = more anomalous).
               Required for ROC-AUC and PR-AUC.

    Returns
    -------
    dict with keys: precision, recall, f1, accuracy,
                    roc_auc, pr_auc, fpr, fnr,
                    tp, fp, fn, tn, n_flows, n_anomalies_true, n_anomalies_pred,
                    roc_curve (fpr_arr, tpr_arr, thresholds),
                    pr_curve  (precision_arr, recall_arr, thresholds)
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

    prec = precision_score(yt, yp, zero_division=0.0)
    rec = recall_score(yt, yp, zero_division=0.0)
    f1 = f1_score(yt, yp, zero_division=0.0)
    acc = accuracy_score(yt, yp)

    fpr_val = fp / max(fp + tn, 1)
    fnr_val = fn / max(fn + tp, 1)

    result = {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "fpr": round(float(fpr_val), 4),
        "fnr": round(float(fnr_val), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_flows": len(yt),
        "n_anomalies_true": int(yt.sum()),
        "n_anomalies_pred": int(yp.sum()),
    }

    if y_scores is not None:
        ys = np.asarray(y_scores, dtype=float)
        n_classes = len(set(yt.tolist()))
        if n_classes >= 2 and not np.all(ys == ys[0]):
            result["roc_auc"] = round(float(roc_auc_score(yt, ys)), 4)

            fpr_arr, tpr_arr, roc_thresh = roc_curve(yt, ys)
            result["roc_curve"] = {
                "fpr": fpr_arr.tolist(),
                "tpr": tpr_arr.tolist(),
                "thresholds": roc_thresh.tolist(),
            }

            prec_arr, rec_arr, pr_thresh = precision_recall_curve(yt, ys)
            result["pr_auc"] = round(float(auc(rec_arr, prec_arr)), 4)
            result["pr_curve"] = {
                "precision": prec_arr.tolist(),
                "recall": rec_arr.tolist(),
                "thresholds": pr_thresh.tolist(),
            }
        else:
            result["roc_auc"] = None
            result["pr_auc"] = None
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    return result
