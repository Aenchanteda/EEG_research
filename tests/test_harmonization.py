import numpy as np
import pytest

from mi_eeg_qa.data.harmonization import (
    BCIC_2A_EEG_CHANNELS,
    common_channel_selection,
    harmonize_binary_labels,
    normalize_channel_name,
)


def test_harmonize_binary_labels_accepts_common_left_right_aliases():
    labels = ["left_hand", "right hand", "T1", "T2"]
    y, classes = harmonize_binary_labels(labels, label_aliases={"t1": "left_hand", "t2": "right_hand"})

    assert classes == ("left_hand", "right_hand")
    np.testing.assert_array_equal(y, np.array([0, 1, 0, 1]))


def test_harmonize_binary_labels_rejects_non_hand_classes():
    with pytest.raises(ValueError, match="unsupported labels"):
        harmonize_binary_labels(["left_hand", "feet"])


def test_common_channel_selection_uses_preferred_bcic_order():
    physionet = ["Fc3.", "FC1.", "FCz.", "C3.", "CPz.", "Oz."]
    bcic = ["Fz", "FC3", "FC1", "FCz", "C3", "CPz"]

    selection = common_channel_selection(physionet, bcic, preferred_order=BCIC_2A_EEG_CHANNELS)

    assert selection.channels == ("FC3", "FC1", "FCz", "C3", "CPz")
    np.testing.assert_array_equal(selection.left_indices, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(selection.right_indices, np.array([1, 2, 3, 4, 5]))


def test_normalize_channel_name_strips_case_and_punctuation():
    assert normalize_channel_name("Fc3.") == "FC3"
    assert normalize_channel_name("cp-z") == "CPZ"
