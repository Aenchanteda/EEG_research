import numpy as np

from mi_eeg_qa.degradation import add_emg_noise, add_eog_artifact, drop_channels
from mi_eeg_qa.sqi import fit_sqi, transform_sqi


def test_degradation_reduces_fitted_sqi_on_average():
    rng = np.random.default_rng(42)
    X_train = rng.normal(scale=0.4, size=(24, 8, 256)).astype(np.float32)
    X_test = rng.normal(scale=0.4, size=(12, 8, 256)).astype(np.float32)

    scorer = fit_sqi(X_train, sfreq=128.0, method="rank")
    q_clean = transform_sqi(scorer, X_test)

    X_degraded = add_eog_artifact(X_test, sfreq=128.0, strength=3.0, probability=1.0, seed=1)
    X_degraded = add_emg_noise(X_degraded, sfreq=128.0, strength=1.5, probability=1.0, seed=2)
    X_degraded, _ = drop_channels(X_degraded, drop_fraction=0.25, seed=3)
    q_degraded = transform_sqi(scorer, X_degraded)

    assert q_degraded.mean() < q_clean.mean()
