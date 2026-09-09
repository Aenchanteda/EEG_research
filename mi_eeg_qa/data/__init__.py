"""Data loaders and synthetic EEG generation."""

from .harmonization import (
    BCIC_2A_EEG_CHANNELS,
    ChannelSelection,
    common_channel_selection,
    harmonize_binary_labels,
    normalize_channel_name,
)
from .synthetic import SyntheticEEGConfig, generate_synthetic_mi

__all__ = [
    "BCIC_2A_EEG_CHANNELS",
    "ChannelSelection",
    "SyntheticEEGConfig",
    "common_channel_selection",
    "generate_synthetic_mi",
    "harmonize_binary_labels",
    "normalize_channel_name",
]
