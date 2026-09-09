#!/usr/bin/env python3
"""Run a deterministic synthetic smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from mi_eeg_qa.abstention import confidence_scores, fusion_score, sqi_accept, threshold_sweep
from mi_eeg_qa.data import SyntheticEEGConfig, generate_synthetic_mi
from mi_eeg_qa.degradation import add_emg_noise, add_eog_artifact, drop_channels
from mi_eeg_qa.metrics import accuracy, brier_score, expected_calibration_error, risk_coverage_curve
from mi_eeg_qa.models import CSPLDAClassifier
from mi_eeg_qa.preprocess import bandpass_epochs, standardize_epochs
from mi_eeg_qa.protocols import deterministic_train_test_split
from mi_eeg_qa.sqi import compute_sqi


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _summarize_policy(name: str, y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    if np.any(mask):
        acc = accuracy(y_true[mask], pred[mask])
        risk = 1.0 - acc
    else:
        acc = float("nan")
        risk = float("nan")
    return {"name": name, "coverage": float(mask.mean()), "accuracy": acc, "risk": risk}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    seed = int(cfg.get("seed", 7))
    np.random.seed(seed)

    synthetic_cfg = SyntheticEEGConfig(seed=seed, **cfg["synthetic"])
    X, y, meta = generate_synthetic_mi(synthetic_cfg)
    X = add_eog_artifact(X, sfreq=meta["sfreq"], strength=1.2, probability=0.2, seed=seed + 1)
    X = add_emg_noise(X, sfreq=meta["sfreq"], strength=0.25, probability=0.2, seed=seed + 2)
    X, dropped = drop_channels(X, drop_fraction=0.06, seed=seed + 3)

    pp = cfg.get("preprocess", {})
    X = bandpass_epochs(X, sfreq=meta["sfreq"], low=float(pp.get("low", 8.0)), high=float(pp.get("high", 30.0)))
    X = standardize_epochs(X)
    q = compute_sqi(X, sfreq=meta["sfreq"])

    X_train, X_test, y_train, y_test, q_train, q_test = deterministic_train_test_split(X, y, q, seed=seed)
    _ = q_train  # Kept for symmetry with future calibration splits.
    model = CSPLDAClassifier(n_components=int(cfg.get("model", {}).get("csp_components", 6)))
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    conf = confidence_scores(proba)
    fused = fusion_score(proba, q_test, weight_confidence=0.55)
    soft_thr = float(cfg.get("abstention", {}).get("softmax_threshold", 0.55))
    sqi_thr = float(cfg.get("abstention", {}).get("sqi_threshold", 0.45))
    policies = [
        _summarize_policy("forced", y_test, proba, np.ones_like(y_test, dtype=bool)),
        _summarize_policy("softmax", y_test, proba, conf >= soft_thr),
        _summarize_policy("sqi", y_test, proba, sqi_accept(q_test, sqi_thr)),
        _summarize_policy("combined_and", y_test, proba, (conf >= soft_thr) & (q_test >= sqi_thr)),
        _summarize_policy("fusion", y_test, proba, fused >= 0.5),
    ]
    thresholds = np.array(cfg.get("abstention", {}).get("thresholds") or np.linspace(0, 1, 21), dtype=float)
    sweeps = {
        "softmax": threshold_sweep(y_test, proba, conf, thresholds),
        "sqi": threshold_sweep(y_test, proba, q_test, thresholds),
        "fusion": threshold_sweep(y_test, proba, fused, thresholds),
    }
    rc = risk_coverage_curve(y_test, proba, confidence=fused)

    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": seed,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "dropped_channels": dropped.tolist(),
        "metrics": {
            "accuracy": accuracy(y_test, proba.argmax(axis=1)),
            "ece_15": expected_calibration_error(y_test, proba, n_bins=15),
            "brier": brier_score(y_test, proba),
        },
        "policies": policies,
        "threshold_sweeps": sweeps,
    }
    json_path = artifacts_dir / "smoke_results.json"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    plt.plot(rc["coverage"], rc["risk"], marker="o", linewidth=1.5)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title("Synthetic MI risk-coverage (fusion score)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = artifacts_dir / "smoke_risk_coverage.png"
    plt.savefig(plot_path, dpi=140)

    print("Synthetic MI smoke experiment complete")
    print(json.dumps(result["metrics"], indent=2))
    print("Policies:")
    for row in policies:
        print(f"  {row['name']}: coverage={row['coverage']:.3f} accuracy={row['accuracy']:.3f} risk={row['risk']:.3f}")
    print(f"Wrote {json_path} and {plot_path}")


if __name__ == "__main__":
    main()
