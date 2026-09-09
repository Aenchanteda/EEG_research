import numpy as np

from mi_eeg_qa.sqi import compute_sqi, compute_sqi_features, fit_sqi, transform_sqi


def test_sqi_is_bounded_and_penalizes_flat_channel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4, 128)).astype(np.float32)
    X[-1, :, :] = 0.0
    q = compute_sqi(X, sfreq=128.0)
    assert q.shape == (8,)
    assert np.all((0.0 <= q) & (q <= 1.0))
    assert q[-1] < np.median(q[:-1])


def test_sqi_features_shapes():
    X = np.ones((3, 2, 64), dtype=np.float32)
    features = compute_sqi_features(X, sfreq=128.0)
    assert features.peak_to_peak_rms.shape == (3,)
    assert features.hf_power_proxy.shape == (3,)
    assert features.flat_channel_fraction.shape == (3,)


def test_fitted_sqi_uses_train_reference_for_test_transform():
    rng = np.random.default_rng(1)
    X_train = rng.normal(scale=1.0, size=(10, 4, 128)).astype(np.float32)
    X_test = rng.normal(scale=1.0, size=(4, 4, 128)).astype(np.float32)
    X_test[-1, :, :] = 0.0

    scorer = fit_sqi(X_train, sfreq=128.0, method="rank")
    q_test = transform_sqi(scorer, X_test)
    q_test_self_ranked = compute_sqi(X_test, sfreq=128.0, method="rank")

    assert q_test.shape == (4,)
    assert np.all((0.0 <= q_test) & (q_test <= 1.0))
    assert not np.allclose(q_test, q_test_self_ranked)
    assert q_test[-1] == 0.0
