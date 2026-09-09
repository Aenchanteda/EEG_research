"""Deterministic split helpers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def deterministic_train_test_split(*arrays: np.ndarray, test_size: float = 0.35, seed: int = 7):
    """Stratified train/test split using the second positional array as labels."""

    if len(arrays) < 2:
        raise ValueError("Provide at least X and y")
    return train_test_split(*arrays, test_size=test_size, random_state=seed, stratify=arrays[1])
