"""Deterministic synthetic motor imagery EEG-like data for smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticEEGConfig:
    n_trials: int = 160
    n_channels: int = 16
    n_times: int = 256
    sfreq: float = 128.0
    n_classes: int = 2
    class_sep: float = 1.2
    noise: float = 0.75
    seed: int = 7


def generate_synthetic_mi(config: SyntheticEEGConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate EEG-like epochs with class-specific mu/beta rhythms.

    Returns
    -------
    X : ndarray, shape (trials, channels, times)
        Synthetic epochs in arbitrary microvolt-like units.
    y : ndarray, shape (trials,)
        Integer class labels.
    meta : dict
        Sampling frequency and channel names.
    """

    cfg = config or SyntheticEEGConfig()
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.n_times) / cfg.sfreq
    y = np.arange(cfg.n_trials, dtype=int) % cfg.n_classes
    rng.shuffle(y)

    X = rng.normal(0.0, cfg.noise, size=(cfg.n_trials, cfg.n_channels, cfg.n_times))
    spatial = rng.normal(0.0, 0.15, size=(cfg.n_classes, cfg.n_channels))

    # Stable class topographies around central channels, where MI rhythms are expected.
    centers = np.linspace(cfg.n_channels * 0.35, cfg.n_channels * 0.65, cfg.n_classes)
    channel_idx = np.arange(cfg.n_channels)
    for cls, center in enumerate(centers):
        spatial[cls] += np.exp(-0.5 * ((channel_idx - center) / 2.0) ** 2)
        spatial[cls] /= np.linalg.norm(spatial[cls]) + 1e-12

    for i, cls in enumerate(y):
        phase = rng.uniform(0, 2 * np.pi)
        mu = np.sin(2 * np.pi * (10.0 + cls) * t + phase)
        beta = 0.45 * np.sin(2 * np.pi * (20.0 + 1.5 * cls) * t + phase / 2)
        envelope = np.hanning(cfg.n_times)
        source = cfg.class_sep * (mu + beta) * envelope
        X[i] += spatial[cls, :, None] * source[None, :]

    channel_names = [f"EEG{i:02d}" for i in range(cfg.n_channels)]
    meta = {"sfreq": cfg.sfreq, "channel_names": channel_names, "description": "synthetic_mi"}
    return X.astype(np.float32), y.astype(int), meta
