import numpy as np

from mi_eeg_qa.metrics import accuracy, accuracy_at_coverage, brier_score, expected_calibration_error, risk_coverage_curve


def test_accuracy_ece_brier_known_values():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
    assert accuracy(y, proba.argmax(axis=1)) == 0.5
    assert np.isclose(brier_score(y, proba), 0.65)
    ece = expected_calibration_error(y, proba, n_bins=2)
    assert 0.0 <= ece <= 1.0


def test_risk_coverage_and_accuracy_at_coverage():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8], [0.6, 0.4]])
    curve = risk_coverage_curve(y, proba)
    assert np.all(np.diff(curve["coverage"]) > 0)
    assert curve["risk"].shape == (4,)
    assert np.isclose(accuracy_at_coverage(y, proba, coverage=0.5), 1.0)
