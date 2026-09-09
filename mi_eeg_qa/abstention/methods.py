"""Abstention scores and accept/reject policies."""

from __future__ import annotations

import numpy as np

from mi_eeg_qa.metrics import accuracy


def temperature_scale(logits_or_proba: np.ndarray, temperature: float = 1.0, input_is_proba: bool = True) -> np.ndarray:
    """Apply temperature scaling and return probabilities."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    x = np.asarray(logits_or_proba, dtype=float)
    logits = np.log(np.clip(x, 1e-12, 1.0)) if input_is_proba else x
    logits = logits / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def confidence_scores(proba: np.ndarray) -> np.ndarray:
    return np.asarray(proba, dtype=float).max(axis=1)


def margin_scores(proba: np.ndarray) -> np.ndarray:
    sorted_p = np.sort(np.asarray(proba, dtype=float), axis=1)
    if sorted_p.shape[1] < 2:
        return np.ones(sorted_p.shape[0])
    return sorted_p[:, -1] - sorted_p[:, -2]


def forced_accept(n_samples: int) -> np.ndarray:
    return np.ones(n_samples, dtype=bool)


def softmax_accept(proba: np.ndarray, threshold: float) -> np.ndarray:
    return confidence_scores(proba) >= threshold


def sqi_accept(q: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(q, dtype=float) >= threshold


def and_accept(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("Provide at least one mask")
    out = np.ones_like(np.asarray(masks[0], dtype=bool), dtype=bool)
    for mask in masks:
        out &= np.asarray(mask, dtype=bool)
    return out


def fusion_score(proba: np.ndarray, q: np.ndarray, weight_confidence: float = 0.5) -> np.ndarray:
    """Fuse classifier confidence and SQI into a single abstention score."""

    if not 0 <= weight_confidence <= 1:
        raise ValueError("weight_confidence must be in [0, 1]")
    conf = confidence_scores(proba)
    q = np.asarray(q, dtype=float)
    return np.clip(weight_confidence * conf + (1.0 - weight_confidence) * q, 0.0, 1.0)


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | None = None) -> list[dict[str, float]]:
    """Evaluate coverage and accepted accuracy across abstention thresholds."""

    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    pred = proba.argmax(axis=1)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        mask = scores >= threshold
        cov = float(mask.mean())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": cov,
                "accuracy": accuracy(y_true[mask], pred[mask]) if np.any(mask) else float("nan"),
                "risk": 1.0 - accuracy(y_true[mask], pred[mask]) if np.any(mask) else float("nan"),
            }
        )
    return rows
