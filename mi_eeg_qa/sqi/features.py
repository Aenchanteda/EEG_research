"""Segment signal quality features mapped to q in [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata


@dataclass(frozen=True)
class SQIFeatures:
    peak_to_peak_rms: np.ndarray
    hf_power_proxy: np.ndarray
    flat_channel_fraction: np.ndarray


def compute_sqi_features(X: np.ndarray, sfreq: float, flat_std_threshold: float = 1e-3, hf_cut_hz: float = 35.0) -> SQIFeatures:
    """Compute per-epoch quality features.

    Higher values indicate more potential artifact for all three returned features.
    """

    X = np.asarray(X, dtype=float)
    rms = np.sqrt(np.mean(X**2, axis=-1))
    p2p = np.ptp(X, axis=-1)
    peak_to_peak_rms = np.median(p2p / (rms + 1e-12), axis=1)

    freqs = np.fft.rfftfreq(X.shape[-1], d=1.0 / sfreq)
    psd = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    total = psd.sum(axis=-1) + 1e-12
    hf_mask = freqs >= hf_cut_hz
    hf_power_proxy = np.median(psd[..., hf_mask].sum(axis=-1) / total, axis=1) if np.any(hf_mask) else np.zeros(X.shape[0])

    flat_channel_fraction = np.mean(np.std(X, axis=-1) < flat_std_threshold, axis=1)
    return SQIFeatures(
        peak_to_peak_rms=peak_to_peak_rms,
        hf_power_proxy=hf_power_proxy,
        flat_channel_fraction=flat_channel_fraction,
    )


def _rank01(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.full_like(values, 0.5, dtype=float)
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


def compute_sqi(X: np.ndarray, sfreq: float, method: str = "rank") -> np.ndarray:
    """Return per-epoch signal quality q in [0, 1].

    The default rank mapping is robust for small smoke-test datasets: high
    peak-to-peak/RMS, high-frequency power, or flat-channel fraction lower q.
    """

    f = compute_sqi_features(X, sfreq=sfreq)
    badness = np.vstack([f.peak_to_peak_rms, f.hf_power_proxy, f.flat_channel_fraction]).T
    if method == "rank":
        ranked = np.column_stack([_rank01(badness[:, j]) for j in range(badness.shape[1])])
        score = 1.0 - np.average(ranked, axis=1, weights=np.array([0.4, 0.35, 0.25]))
    elif method == "logistic":
        center = np.median(badness, axis=0)
        scale = np.median(np.abs(badness - center), axis=0) + 1e-6
        z = (badness - center) / scale
        score = expit(-np.average(z, axis=1, weights=np.array([0.4, 0.35, 0.25])))
    else:
        raise ValueError("method must be 'rank' or 'logistic'")
    score *= 1.0 - f.flat_channel_fraction
    return np.clip(score, 0.0, 1.0)
