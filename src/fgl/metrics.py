"""Identification and verification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def rank_k(scores: np.ndarray, true_idx: np.ndarray, ks=(1, 5)) -> dict[str, float]:
    """Rank-k identification accuracy from a (probes, classes) score matrix."""
    true_scores = scores[np.arange(len(scores)), true_idx]
    rank = (scores > true_scores[:, None]).sum(axis=1)  # 0 = correct class ranked first
    return {f"rank{k}": float(np.mean(rank < k)) for k in ks}


def class_scores(sim: np.ndarray, gallery_labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Collapse a (probes, gallery samples) similarity matrix to per-class max scores."""
    out = np.full((sim.shape[0], len(classes)), -np.inf)
    for j, c in enumerate(classes):
        cols = gallery_labels == c
        if cols.any():
            out[:, j] = sim[:, cols].max(axis=1)
    return out


def verification(genuine: np.ndarray, impostor: np.ndarray,
                 fars=(1e-2, 1e-3)) -> dict[str, float]:
    """EER, ROC AUC and TAR at fixed FARs from genuine/impostor similarity scores."""
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    out = {"eer": float((fpr[i] + fnr[i]) / 2), "auc": float(roc_auc_score(y, s))}
    for far in fars:
        out[f"tar@far{far:g}"] = float(np.interp(far, fpr, tpr))
    return out
