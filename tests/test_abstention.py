import numpy as np

from mi_eeg_qa.abstention import (
    and_accept,
    confidence_scores,
    forced_accept,
    fusion_score,
    margin_scores,
    softmax_accept,
    sqi_accept,
    temperature_scale,
    threshold_sweep,
)


def test_abstention_scores_and_masks():
    proba = np.array([[0.8, 0.2], [0.51, 0.49], [0.1, 0.9]])
    q = np.array([0.9, 0.2, 0.7])
    assert np.array_equal(forced_accept(3), np.array([True, True, True]))
    assert np.array_equal(softmax_accept(proba, 0.75), np.array([True, False, True]))
    assert np.array_equal(sqi_accept(q, 0.5), np.array([True, False, True]))
    assert np.array_equal(and_accept(softmax_accept(proba, 0.75), sqi_accept(q, 0.5)), np.array([True, False, True]))
    assert np.allclose(confidence_scores(proba), [0.8, 0.51, 0.9])
    assert np.allclose(margin_scores(proba), [0.6, 0.02, 0.8])
    fused = fusion_score(proba, q, weight_confidence=0.5)
    assert np.all((0.0 <= fused) & (fused <= 1.0))


def test_temperature_and_threshold_sweep():
    proba = np.array([[0.8, 0.2], [0.6, 0.4]])
    scaled = temperature_scale(proba, temperature=2.0)
    assert scaled.shape == proba.shape
    assert np.allclose(scaled.sum(axis=1), 1.0)
    rows = threshold_sweep(np.array([0, 1]), proba, confidence_scores(proba), thresholds=np.array([0.0, 0.7]))
    assert rows[0]["coverage"] == 1.0
    assert rows[1]["coverage"] == 0.5
