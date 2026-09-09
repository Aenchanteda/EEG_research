#!/usr/bin/env python3
"""Run BCIC within-subject CSP-LDA experiments through MOABB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder

from mi_eeg_qa.abstention import confidence_scores, fusion_score, margin_scores, threshold_sweep
from mi_eeg_qa.data.moabb_loaders import MOABBRequest, load_moabb_epochs
from mi_eeg_qa.metrics import (
    accuracy,
    accuracy_at_coverage,
    brier_score,
    expected_calibration_error,
    risk_coverage_curve,
)
from mi_eeg_qa.models import CSPLDAClassifier
from mi_eeg_qa.preprocess import bandpass_epochs, standardize_epochs
from mi_eeg_qa.sqi import fit_sqi, transform_sqi


POLICIES = ("forced", "softmax", "margin", "sqi", "combined_and", "fusion")

DATASET_LABELS = {
    "BNCI2014_001": "BCIC IV 2a",
    "BNCI2014_004": "BCIC IV 2b",
}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _ordered_unique(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values.astype(str):
        if value not in out:
            out.append(value)
    return out


def _is_train_session(name: str) -> bool:
    lowered = name.lower()
    return "train" in lowered or lowered.endswith("_t") or lowered.endswith("-t") or lowered.endswith(" t") or lowered.endswith("t")


def _is_test_session(name: str) -> bool:
    lowered = name.lower()
    return (
        "test" in lowered
        or "eval" in lowered
        or lowered.endswith("_e")
        or lowered.endswith("-e")
        or lowered.endswith(" e")
        or lowered.endswith("e")
    )


def _session_split_indices(metadata: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if metadata is None or not hasattr(metadata, "columns") or "session" not in metadata.columns:
        return None
    sessions = np.asarray(metadata["session"].astype(str))
    unique = _ordered_unique(sessions)
    if len(unique) < 2:
        return None

    train_sessions = [s for s in unique if _is_train_session(s)]
    test_sessions = [s for s in unique if _is_test_session(s)]
    if not train_sessions or not test_sessions:
        train_sessions = unique[:-1]
        test_sessions = unique[-1:]

    train_mask = np.isin(sessions, train_sessions)
    test_mask = np.isin(sessions, test_sessions)
    if not np.any(train_mask) or not np.any(test_mask):
        return None
    info = {
        "method": "session",
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
    }
    return np.where(train_mask)[0], np.where(test_mask)[0], info


def _fallback_split_indices(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
    return train_idx, test_idx, {"method": "stratified_holdout", "test_size": float(test_size), "seed": int(seed)}


def _policy_summary(y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    coverage = float(mask.mean())
    accepted_accuracy = accuracy(y_true[mask], pred[mask]) if np.any(mask) else float("nan")
    return {
        "coverage": coverage,
        "accuracy": accepted_accuracy,
        "risk": 1.0 - accepted_accuracy if np.any(mask) else float("nan"),
    }


def _risk_coverage_lists(y_true: np.ndarray, proba: np.ndarray, score: np.ndarray) -> dict[str, list[float]]:
    curve = risk_coverage_curve(y_true, proba, confidence=score)
    return {k: v.tolist() for k, v in curve.items()}


def _evaluate_policies(
    y_true: np.ndarray,
    proba: np.ndarray,
    q: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    abst = cfg.get("abstention", {})
    thresholds = np.asarray(abst.get("thresholds") or np.linspace(0.0, 1.0, 21), dtype=float)
    soft_thr = float(abst.get("softmax_threshold", 0.6))
    margin_thr = float(abst.get("margin_threshold", 0.2))
    sqi_thr = float(abst.get("sqi_threshold", 0.45))
    fusion_thr = float(abst.get("fusion_threshold", 0.5))
    fusion_weight = float(abst.get("fusion_weight_confidence", 0.5))
    fusion_grid = [float(x) for x in abst.get("fusion_weight_grid", [0.25, 0.5, 0.75])]

    conf = confidence_scores(proba)
    margin = margin_scores(proba)
    combined_score = np.minimum(conf, q)
    fused = fusion_score(proba, q, weight_confidence=fusion_weight)

    definitions = {
        "forced": {"score": np.ones_like(conf), "mask": np.ones_like(conf, dtype=bool), "threshold": None},
        "softmax": {"score": conf, "mask": conf >= soft_thr, "threshold": soft_thr},
        "margin": {"score": margin, "mask": margin >= margin_thr, "threshold": margin_thr},
        "sqi": {"score": q, "mask": q >= sqi_thr, "threshold": sqi_thr},
        "combined_and": {
            "score": combined_score,
            "mask": (conf >= soft_thr) & (q >= sqi_thr),
            "threshold": {"softmax": soft_thr, "sqi": sqi_thr},
        },
        "fusion": {"score": fused, "mask": fused >= fusion_thr, "threshold": fusion_thr},
    }

    policy_results: dict[str, Any] = {}
    for name in POLICIES:
        score = definitions[name]["score"]
        policy_results[name] = {
            "default_threshold": definitions[name]["threshold"],
            "default": _policy_summary(y_true, proba, definitions[name]["mask"]),
            "accuracy_at_0.8": accuracy_at_coverage(y_true, proba, coverage=0.8, confidence=score),
            "accuracy_at_0.9": accuracy_at_coverage(y_true, proba, coverage=0.9, confidence=score),
            "threshold_sweep": threshold_sweep(y_true, proba, score, thresholds=thresholds),
            "risk_coverage_curve": _risk_coverage_lists(y_true, proba, score),
        }

    policy_results["fusion"]["weight_confidence"] = fusion_weight
    policy_results["fusion_weight_grid"] = {
        f"{weight:.2f}": {
            "threshold_sweep": threshold_sweep(
                y_true,
                proba,
                fusion_score(proba, q, weight_confidence=weight),
                thresholds=thresholds,
            ),
            "risk_coverage_curve": _risk_coverage_lists(
                y_true,
                proba,
                fusion_score(proba, q, weight_confidence=weight),
            ),
        }
        for weight in fusion_grid
    }
    return policy_results


def _dataset_label(cfg: dict[str, Any]) -> str:
    ds = cfg.get("dataset", {})
    return str(ds.get("label") or DATASET_LABELS.get(ds.get("name"), ds.get("name", "MOABB dataset")))


def _run_subject(subject: int, cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    seed = int(cfg.get("seed", 7))
    ds = cfg["dataset"]
    dataset_name = str(ds.get("name", "BNCI2014_001"))
    request = MOABBRequest(dataset=dataset_name, subjects=(subject,), paradigm=ds.get("paradigm", "left_right_hand"))
    X, y_raw, metadata = load_moabb_epochs(request)
    X = np.asarray(X, dtype=np.float32)

    label_encoder = LabelEncoder().fit(y_raw)
    y = label_encoder.transform(y_raw)
    if len(label_encoder.classes_) != 2:
        raise ValueError(
            f"{dataset_name} CSP-LDA run requires binary labels; got {list(label_encoder.classes_)}"
        )

    split_cfg = cfg.get("split", {})
    split = None
    if bool(split_cfg.get("prefer_sessions", True)):
        split = _session_split_indices(metadata)
    if split is None:
        split = _fallback_split_indices(y, test_size=float(split_cfg.get("fallback_test_size", 0.35)), seed=seed)
    train_idx, test_idx, split_info = split

    pp = cfg.get("preprocess", {})
    sfreq = float(ds.get("sfreq", 250.0))
    X = bandpass_epochs(X, sfreq=sfreq, low=float(pp.get("low", 8.0)), high=float(pp.get("high", 30.0)))
    X = standardize_epochs(X)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    sqi_cfg = cfg.get("sqi", {})
    sqi_scorer = fit_sqi(X_train, sfreq=sfreq, method=sqi_cfg.get("method", "rank"))
    q_train = transform_sqi(sqi_scorer, X_train)
    q_test = transform_sqi(sqi_scorer, X_test)

    model_cfg = cfg.get("model", {})
    if model_cfg.get("name", "csp_lda") != "csp_lda":
        raise ValueError("This within-subject script currently supports model.name: csp_lda")
    model = CSPLDAClassifier(n_components=int(model_cfg.get("csp_components", 8)))
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy(y_test, proba.argmax(axis=1)),
        "expected_calibration_error": expected_calibration_error(
            y_test,
            proba,
            n_bins=int(cfg.get("abstention", {}).get("ece_bins", 15)),
        ),
        "brier_score": brier_score(y_test, proba),
    }
    policies = _evaluate_policies(y_test, proba, q_test, cfg)

    result = {
        "dataset": dataset_name,
        "dataset_label": _dataset_label(cfg),
        "subject": int(subject),
        "labels": label_encoder.classes_.tolist(),
        "seed": seed,
        "sfreq": sfreq,
        "split": split_info,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "sqi": {
            "method": sqi_cfg.get("method", "rank"),
            "train_mean": float(np.mean(q_train)),
            "test_mean": float(np.mean(q_test)),
        },
        "metrics": metrics,
        "policies": policies,
        "_y_test": y_test,
        "_proba": proba,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_json = output_dir / f"subject_{subject:02d}.json"
    public_result = {k: v for k, v in result.items() if not k.startswith("_")}
    subject_json.write_text(json.dumps(_jsonable(public_result), indent=2, allow_nan=True), encoding="utf-8")
    _plot_subject(result, output_dir / f"subject_{subject:02d}_risk_coverage.png")
    return result


def _plot_subject(result: dict[str, Any], path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for name in POLICIES:
        curve = result["policies"][name]["risk_coverage_curve"]
        plt.plot(curve["coverage"], curve["risk"], label=name, linewidth=1.2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title(f"{result.get('dataset_label', result['dataset'])} subject {result['subject']:02d} risk-coverage")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _write_summary(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_subjects": len(results), "subjects": [int(r["subject"]) for r in results]}
    metric_names = ("accuracy", "expected_calibration_error", "brier_score")
    summary["metrics_mean"] = {
        name: float(np.mean([r["metrics"][name] for r in results])) for name in metric_names
    }
    y_all = np.concatenate([np.asarray(r["_y_test"]) for r in results])
    proba_all = np.concatenate([np.asarray(r["_proba"]) for r in results])
    summary["metrics_pooled"] = {
        "accuracy": accuracy(y_all, proba_all.argmax(axis=1)),
        "expected_calibration_error": expected_calibration_error(y_all, proba_all),
        "brier_score": brier_score(y_all, proba_all),
        "n_trials": int(y_all.size),
    }
    summary["policies_mean"] = {}
    for policy in POLICIES:
        summary["policies_mean"][policy] = {
            key: float(np.nanmean([r["policies"][policy]["default"][key] for r in results]))
            for key in ("coverage", "accuracy", "risk")
        }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=True), encoding="utf-8")

    plt.figure(figsize=(7, 4.5))
    for result in results:
        curve = result["policies"]["fusion"]["risk_coverage_curve"]
        plt.plot(curve["coverage"], curve["risk"], alpha=0.35, linewidth=1.0, label=f"S{result['subject']:02d}")
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    title_label = results[0].get("dataset_label", results[0]["dataset"]) if results else "BCIC"
    plt.title(f"{title_label} fusion risk-coverage by subject")
    plt.grid(True, alpha=0.3)
    if len(results) <= 9:
        plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_fusion_risk_coverage.png", dpi=140)
    plt.close()
    return summary


def _fmt_float(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.3f}"


def _write_run_notes(
    cfg: dict[str, Any],
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    command: str,
) -> None:
    ds = cfg.get("dataset", {})
    pp = cfg.get("preprocess", {})
    model = cfg.get("model", {})
    dataset_name = str(ds.get("name", results[0]["dataset"] if results else "MOABB"))
    dataset_label = _dataset_label(cfg)
    channel_note = str(ds.get("channel_note", "MOABB paradigm channel selection"))
    split_methods = sorted({str(r["split"]["method"]) for r in results})
    pooled = summary["metrics_pooled"]
    mean = summary["metrics_mean"]

    lines = [
        f"# {dataset_label} within-subject CSP-LDA abstention run",
        "",
        f"- Command: `{command}`",
        f"- Config: `{cfg.get('_config_path', 'unknown')}`",
        f"- Output directory: `{output_dir.as_posix()}/`",
        f"- Dataset: MOABB `{dataset_name}`, `{ds.get('paradigm', 'left_right_hand')}` paradigm",
        f"- Subjects completed: {', '.join(str(r['subject']) for r in results)}",
        f"- Bandpass: {float(pp.get('low', 8.0)):.1f}-{float(pp.get('high', 30.0)):.1f} Hz",
        f"- Model: `{model.get('name', 'csp_lda')}` with {int(model.get('csp_components', 8))} CSP components",
        f"- Channel montage note: {channel_note}",
        f"- SQI: fit on train split only, then transformed train/test with method `{cfg.get('sqi', {}).get('method', 'rank')}`",
        f"- Split methods observed: {', '.join(split_methods)}",
        "- Failures/errors: none.",
        "",
        "## Pooled held-out metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {_fmt_float(float(pooled['accuracy']))} |",
        f"| ECE | {_fmt_float(float(pooled['expected_calibration_error']))} |",
        f"| Brier | {_fmt_float(float(pooled['brier_score']))} |",
        f"| Trials | {int(pooled['n_trials'])} |",
        "",
        "## Subject-mean metrics",
        "",
        "| Metric | Mean |",
        "| --- | ---: |",
        f"| Accuracy | {_fmt_float(float(mean['accuracy']))} |",
        f"| ECE | {_fmt_float(float(mean['expected_calibration_error']))} |",
        f"| Brier | {_fmt_float(float(mean['brier_score']))} |",
        "",
        "## Default policy means",
        "",
        "| Policy | Coverage | Accepted accuracy | Risk |",
        "| --- | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        row = summary["policies_mean"][policy]
        lines.append(
            f"| {policy} | {_fmt_float(float(row['coverage']))} | "
            f"{_fmt_float(float(row['accuracy']))} | {_fmt_float(float(row['risk']))} |"
        )

    lines.extend(
        [
            "",
            "## Per-subject compact metrics",
            "",
            "| Subject | Accuracy | ECE | Brier | Forced cov | Softmax cov/acc | Margin cov/acc | SQI cov/acc | Combined cov/acc | Fusion cov/acc | Split |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        metrics = result["metrics"]
        policies = result["policies"]
        lines.append(
            f"| {int(result['subject'])} | {_fmt_float(float(metrics['accuracy']))} | "
            f"{_fmt_float(float(metrics['expected_calibration_error']))} | "
            f"{_fmt_float(float(metrics['brier_score']))} | "
            f"{_fmt_float(float(policies['forced']['default']['coverage']))} | "
            f"{_fmt_float(float(policies['softmax']['default']['coverage']))}/"
            f"{_fmt_float(float(policies['softmax']['default']['accuracy']))} | "
            f"{_fmt_float(float(policies['margin']['default']['coverage']))}/"
            f"{_fmt_float(float(policies['margin']['default']['accuracy']))} | "
            f"{_fmt_float(float(policies['sqi']['default']['coverage']))}/"
            f"{_fmt_float(float(policies['sqi']['default']['accuracy']))} | "
            f"{_fmt_float(float(policies['combined_and']['default']['coverage']))}/"
            f"{_fmt_float(float(policies['combined_and']['default']['accuracy']))} | "
            f"{_fmt_float(float(policies['fusion']['default']['coverage']))}/"
            f"{_fmt_float(float(policies['fusion']['default']['accuracy']))} | "
            f"{result['split']['method']} |"
        )

    lines.extend(
        [
            "",
            "## Artifact inventory",
            "",
            "- `summary.json`",
            "- `summary_fusion_risk_coverage.png`",
        ]
    )
    for result in results:
        subject = int(result["subject"])
        lines.append(f"- `subject_{subject:02d}.json`, `subject_{subject:02d}_risk_coverage.png`")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RUN_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _subjects_from_args(args: argparse.Namespace, cfg: dict[str, Any]) -> list[int]:
    if args.subject:
        subjects = args.subject
    else:
        subjects = list(cfg.get("dataset", {}).get("subjects", list(range(1, 10))))
    if args.smoke_moabb:
        return [int(subjects[0])]
    return [int(s) for s in subjects]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BCIC within-subject CSP-LDA experiments.")
    parser.add_argument("--config", default="configs/bcic2a.yaml")
    parser.add_argument("--subject", type=int, action="append", help="Subject to run; can be repeated. Overrides config list.")
    parser.add_argument("--smoke-moabb", action="store_true", help="Run only one configured subject to validate MOABB download/loading.")
    parser.add_argument("--output-dir", default=None, help="Override artifact output directory.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    cfg["_config_path"] = args.config
    output_dir = Path(args.output_dir or cfg.get("artifacts_dir", "artifacts/bcic2a"))
    subjects = _subjects_from_args(args, cfg)

    try:
        results = []
        for subject in subjects:
            dataset_name = cfg.get("dataset", {}).get("name", "BNCI2014_001")
            print(f"Running {dataset_name} within-subject CSP-LDA for subject {subject:02d}")
            result = _run_subject(subject, cfg, output_dir)
            results.append(result)
            m = result["metrics"]
            forced = result["policies"]["forced"]["default"]
            fusion = result["policies"]["fusion"]["default"]
            print(
                f"  accuracy={m['accuracy']:.3f} ece={m['expected_calibration_error']:.3f} "
                f"brier={m['brier_score']:.3f} forced_cov={forced['coverage']:.3f} "
                f"fusion_cov={fusion['coverage']:.3f} fusion_acc={fusion['accuracy']:.3f}"
            )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    summary = _write_summary(results, output_dir)
    _write_run_notes(cfg, results, summary, output_dir, "python3 " + " ".join(sys.argv))
    print(f"Wrote per-subject JSON/PNG artifacts and summary under {output_dir}")
    print(json.dumps(_jsonable(summary["metrics_pooled"]), indent=2))


if __name__ == "__main__":
    main()
