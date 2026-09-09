"""Metrics for accuracy, calibration, and risk-coverage."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-12)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    one_hot = np.zeros_like(proba, dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def risk_coverage_curve(y_true: np.ndarray, proba: np.ndarray, confidence: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Compute cumulative risk after accepting samples from high to low confidence."""

    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    conf = np.asarray(confidence, dtype=float) if confidence is not None else proba.max(axis=1)
    pred = proba.argmax(axis=1)
    order = np.argsort(-conf, kind="mergesort")
    correct = (pred[order] == y_true[order]).astype(float)
    accepted = np.arange(1, len(y_true) + 1)
    coverage = accepted / len(y_true)
    risk = 1.0 - np.cumsum(correct) / accepted
    return {"coverage": coverage, "risk": risk, "confidence": conf[order]}


def accuracy_at_coverage(y_true: np.ndarray, proba: np.ndarray, coverage: float, confidence: np.ndarray | None = None) -> float:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    conf = np.asarray(confidence, dtype=float) if confidence is not None else proba.max(axis=1)
    k = max(1, int(np.ceil(len(y_true) * coverage)))
    order = np.argsort(-conf, kind="mergesort")[:k]
    return accuracy(y_true[order], proba.argmax(axis=1)[order])
