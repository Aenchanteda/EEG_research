"""Small preprocessing helpers for EEG epochs."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_epochs(X: np.ndarray, sfreq: float, low: float = 8.0, high: float = 30.0, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass to epochs."""

    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def standardize_epochs(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each epoch/channel over time."""

    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return ((X - mean) / (std + eps)).astype(np.float32)
