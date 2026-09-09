"""Dataset harmonization helpers for cross-dataset MI experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


LEFT_LABELS = {
    "left",
    "left_hand",
    "left hand",
    "left_hand_mi",
    "left hand mi",
    "left fist",
    "left_fist",
    "imagined_left_hand",
    "imagery_left_hand",
}
RIGHT_LABELS = {
    "right",
    "right_hand",
    "right hand",
    "right_hand_mi",
    "right hand mi",
    "right fist",
    "right_fist",
    "imagined_right_hand",
    "imagery_right_hand",
}


BCIC_2A_EEG_CHANNELS = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)


@dataclass(frozen=True)
class ChannelSelection:
    """Common channel order and selected indices for two datasets."""

    channels: tuple[str, ...]
    left_indices: np.ndarray
    right_indices: np.ndarray


def normalize_channel_name(name: str) -> str:
    """Return a case-insensitive 10-20 channel key without punctuation."""

    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def canonical_channel_name(name: str) -> str:
    """Format a channel name with conventional z suffix casing."""

    cleaned = normalize_channel_name(name)
    if cleaned.endswith("Z"):
        return cleaned[:-1] + "z"
    return cleaned


def _channel_lookup(ch_names: Sequence[str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, name in enumerate(ch_names):
        key = normalize_channel_name(name)
        lookup.setdefault(key, idx)
    return lookup


def common_channel_selection(
    left_ch_names: Sequence[str],
    right_ch_names: Sequence[str],
    preferred_order: Sequence[str] | None = None,
) -> ChannelSelection:
    """Select shared EEG channels in a deterministic order.

    If ``preferred_order`` is supplied, only channels in that order are kept.
    Otherwise the left dataset order is used for all channels present in both
    datasets. Returned indices select channels from the original arrays.
    """

    left_lookup = _channel_lookup(left_ch_names)
    right_lookup = _channel_lookup(right_ch_names)
    if preferred_order is None:
        keys = [normalize_channel_name(ch) for ch in left_ch_names if normalize_channel_name(ch) in right_lookup]
    else:
        keys = [
            normalize_channel_name(ch)
            for ch in preferred_order
            if normalize_channel_name(ch) in left_lookup and normalize_channel_name(ch) in right_lookup
        ]
    if not keys:
        raise ValueError("No shared channels found for cross-dataset harmonization")
    channels = tuple(canonical_channel_name(key) for key in keys)
    return ChannelSelection(
        channels=channels,
        left_indices=np.asarray([left_lookup[key] for key in keys], dtype=int),
        right_indices=np.asarray([right_lookup[key] for key in keys], dtype=int),
    )


def harmonize_binary_labels(
    labels: Iterable[object],
    label_aliases: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, tuple[str, str]]:
    """Map left/right hand labels to integer classes 0/1.

    Unknown labels raise ``ValueError`` so callers do not silently mix in feet,
    rest, or executed-movement classes.
    """

    aliases = {str(k).lower(): str(v).lower() for k, v in (label_aliases or {}).items()}
    y: list[int] = []
    bad: list[str] = []
    for raw in labels:
        text = str(raw).strip().lower().replace("-", "_")
        text = aliases.get(text, text)
        collapsed = text.replace("_", " ")
        if text in LEFT_LABELS or collapsed in LEFT_LABELS:
            y.append(0)
        elif text in RIGHT_LABELS or collapsed in RIGHT_LABELS:
            y.append(1)
        else:
            bad.append(str(raw))
    if bad:
        unique = sorted(set(bad))
        raise ValueError(f"Expected only left/right hand labels; got unsupported labels: {unique}")
    return np.asarray(y, dtype=int), ("left_hand", "right_hand")
