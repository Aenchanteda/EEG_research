"""Helpers to inject EOG/EMG/channel-drop degradation into epochs."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def add_eog_artifact(X: np.ndarray, sfreq: float, strength: float = 2.0, probability: float = 0.3, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.array(X, copy=True)
    n_trials, n_channels, n_times = out.shape
    affected = rng.random(n_trials) < probability
    t = np.arange(n_times) / sfreq
    blink = np.exp(-0.5 * ((t - t.mean()) / 0.08) ** 2)
    frontal_weights = np.linspace(1.0, 0.2, n_channels)
    for i in np.where(affected)[0]:
        out[i] += strength * rng.uniform(0.7, 1.3) * frontal_weights[:, None] * blink[None, :]
    return out.astype(np.float32)


def add_emg_noise(X: np.ndarray, sfreq: float, strength: float = 0.6, probability: float = 0.3, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.array(X, copy=True)
    n_trials, _, _ = out.shape
    affected = rng.random(n_trials) < probability
    high = min(0.45 * sfreq, 55.0)
    low = min(35.0, high * 0.8)
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    for i in np.where(affected)[0]:
        noise = rng.normal(size=out[i].shape)
        out[i] += strength * sosfiltfilt(sos, noise, axis=-1)
    return out.astype(np.float32)


def drop_channels(X: np.ndarray, drop_fraction: float = 0.1, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= drop_fraction <= 1:
        raise ValueError("drop_fraction must be in [0, 1]")
    rng = np.random.default_rng(seed)
    out = np.array(X, copy=True)
    n_drop = int(round(out.shape[1] * drop_fraction))
    channels = np.array([], dtype=int) if n_drop == 0 else np.sort(rng.choice(out.shape[1], size=n_drop, replace=False))
    out[:, channels, :] = 0.0
    return out.astype(np.float32), channels
