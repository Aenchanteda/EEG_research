#!/usr/bin/env python3
"""Run controlled BCIC IV 2a test-time artifact degradation experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder

from mi_eeg_qa.abstention import confidence_scores, fusion_score, margin_scores, threshold_sweep
from mi_eeg_qa.data.moabb_loaders import MOABBRequest, load_moabb_epochs
from mi_eeg_qa.degradation import add_emg_noise, add_eog_artifact, drop_channels
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
KEY_POLICIES = ("softmax", "sqi", "fusion")


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


def _forced_metrics(y_true: np.ndarray, proba: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, proba.argmax(axis=1)),
        "expected_calibration_error": expected_calibration_error(
            y_true,
            proba,
            n_bins=int(cfg.get("abstention", {}).get("ece_bins", 15)),
        ),
        "brier_score": brier_score(y_true, proba),
    }


def _degrade_test_epochs(
    X_test: np.ndarray,
    artifact: str,
    severity: str,
    cfg: dict[str, Any],
    sfreq: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if artifact == "clean" or severity == "none":
        return np.array(X_test, copy=True), {"type": "clean", "severity": "none", "parameters": {}}

    degradation_cfg = cfg.get("degradation", {})
    artifact_cfg = degradation_cfg.get(artifact, {})
    if severity not in artifact_cfg:
        raise ValueError(f"No degradation config for artifact={artifact!r}, severity={severity!r}")
    params = dict(artifact_cfg[severity])
    if artifact == "eog":
        X_degraded = add_eog_artifact(
            X_test,
            sfreq=sfreq,
            strength=float(params.get("strength", 1.0)),
            probability=float(params.get("probability", 0.3)),
            seed=seed,
        )
        extras: dict[str, Any] = {}
    elif artifact == "emg":
        X_degraded = add_emg_noise(
            X_test,
            sfreq=sfreq,
            strength=float(params.get("strength", 0.5)),
            probability=float(params.get("probability", 0.3)),
            seed=seed,
        )
        extras = {}
    elif artifact == "drop":
        X_degraded, channels = drop_channels(
            X_test,
            drop_fraction=float(params.get("drop_fraction", 0.1)),
            seed=seed,
        )
        extras = {"dropped_channels": channels.tolist()}
    else:
        raise ValueError(f"Unsupported degradation artifact {artifact!r}")

    return X_degraded, {"type": artifact, "severity": severity, "parameters": params, **extras}


def _condition_key(artifact: str, severity: str) -> str:
    return "clean" if artifact == "clean" or severity == "none" else f"{artifact}_{severity}"


def _condition_label(condition: dict[str, Any]) -> str:
    artifact = condition["artifact"]
    severity = condition["severity"]
    return "clean" if artifact == "clean" else f"{artifact}/{severity}"


def _condition_specs(cfg: dict[str, Any], selected_artifacts: list[str] | None, selected_severities: list[str] | None) -> list[tuple[str, str]]:
    degradation_cfg = cfg.get("degradation", {})
    artifacts = selected_artifacts or list(degradation_cfg.get("artifacts", ["eog", "emg", "drop"]))
    severities = selected_severities or list(degradation_cfg.get("severities", ["none", "low", "mid", "high"]))

    specs: list[tuple[str, str]] = [("clean", "none")]
    for artifact in artifacts:
        for severity in severities:
            if severity == "none":
                continue
            specs.append((artifact, severity))
    return specs


def _load_subject_data(subject: int, cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 7))
    ds = cfg["dataset"]
    request = MOABBRequest(dataset="BNCI2014_001", subjects=(subject,), paradigm=ds.get("paradigm", "left_right_hand"))
    X, y_raw, metadata = load_moabb_epochs(request)
    X = np.asarray(X, dtype=np.float32)

    label_encoder = LabelEncoder().fit(y_raw)
    y = label_encoder.transform(y_raw)
    if len(label_encoder.classes_) != 2:
        raise ValueError(
            f"BCIC IV 2a CSP-LDA run requires binary left_right_hand labels; got {list(label_encoder.classes_)}"
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

    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "labels": label_encoder.classes_.tolist(),
        "sfreq": sfreq,
        "split": split_info,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def _run_subject(
    subject: int,
    cfg: dict[str, Any],
    output_dir: Path,
    condition_specs: list[tuple[str, str]],
) -> dict[str, Any]:
    seed = int(cfg.get("seed", 7))
    data = _load_subject_data(subject, cfg)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    sfreq = float(data["sfreq"])

    sqi_cfg = cfg.get("sqi", {})
    sqi_scorer = fit_sqi(X_train, sfreq=sfreq, method=sqi_cfg.get("method", "rank"))
    q_train = transform_sqi(sqi_scorer, X_train)

    model_cfg = cfg.get("model", {})
    if model_cfg.get("name", "csp_lda") != "csp_lda":
        raise ValueError("This BCIC IV 2a degradation script currently supports model.name: csp_lda")
    model = CSPLDAClassifier(n_components=int(model_cfg.get("csp_components", 8)))
    model.fit(X_train, y_train)

    conditions: dict[str, Any] = {}
    for offset, (artifact, severity) in enumerate(condition_specs):
        condition_seed = seed + 1000 * int(subject) + offset
        X_eval, degradation_info = _degrade_test_epochs(
            X_test,
            artifact=artifact,
            severity=severity,
            cfg=cfg,
            sfreq=sfreq,
            seed=condition_seed,
        )
        q_test = transform_sqi(sqi_scorer, X_eval)
        proba = model.predict_proba(X_eval)
        key = _condition_key(artifact, severity)
        conditions[key] = {
            "artifact": "clean" if artifact == "clean" else artifact,
            "severity": "none" if artifact == "clean" else severity,
            "seed": condition_seed,
            "degradation": degradation_info,
            "sqi": {
                "test_mean": float(np.mean(q_test)),
                "test_median": float(np.median(q_test)),
            },
            "metrics": _forced_metrics(y_test, proba, cfg),
            "policies": _evaluate_policies(y_test, proba, q_test, cfg),
        }

    result = {
        "dataset": "BNCI2014_001",
        "subject": int(subject),
        "labels": data["labels"],
        "seed": seed,
        "sfreq": sfreq,
        "split": data["split"],
        "n_train": data["n_train"],
        "n_test": data["n_test"],
        "protocol": "CSP-LDA and SQI fit on clean train; degradation injected on test epochs only.",
        "sqi_train": {
            "method": sqi_cfg.get("method", "rank"),
            "train_mean": float(np.mean(q_train)),
            "train_median": float(np.median(q_train)),
        },
        "conditions": conditions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_json = output_dir / f"subject_{subject:02d}.json"
    subject_json.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=True), encoding="utf-8")
    _plot_subject_mid_curves(result, output_dir / f"subject_{subject:02d}_mid_risk_coverage.png")
    return result


def _condition_order(results: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for result in results:
        for key in result["conditions"].keys():
            if key not in ordered:
                ordered.append(key)
    return ordered


def _write_summary(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    condition_order = _condition_order(results)
    summary: dict[str, Any] = {
        "n_subjects": len(results),
        "subjects": [int(r["subject"]) for r in results],
        "conditions": {},
        "mid_severity_key_table": [],
    }
    metric_names = ("accuracy", "expected_calibration_error", "brier_score")
    for condition_key in condition_order:
        present = [r["conditions"][condition_key] for r in results if condition_key in r["conditions"]]
        first = present[0]
        condition_summary: dict[str, Any] = {
            "artifact": first["artifact"],
            "severity": first["severity"],
            "n_subjects": len(present),
            "metrics_mean": {
                name: float(np.mean([c["metrics"][name] for c in present])) for name in metric_names
            },
            "sqi_mean": float(np.mean([c["sqi"]["test_mean"] for c in present])),
            "policies_mean": {},
        }
        for policy in POLICIES:
            condition_summary["policies_mean"][policy] = {
                key: float(np.nanmean([c["policies"][policy]["default"][key] for c in present]))
                for key in ("coverage", "accuracy", "risk")
            }
        summary["conditions"][condition_key] = condition_summary

        if condition_key == "clean" or first["severity"] == "mid":
            row: dict[str, Any] = {
                "condition": condition_key,
                "artifact": first["artifact"],
                "severity": first["severity"],
                "forced_accuracy": condition_summary["metrics_mean"]["accuracy"],
                "sqi_mean": condition_summary["sqi_mean"],
            }
            for policy in KEY_POLICIES:
                row[f"{policy}_coverage"] = condition_summary["policies_mean"][policy]["coverage"]
                row[f"{policy}_accepted_accuracy"] = condition_summary["policies_mean"][policy]["accuracy"]
            summary["mid_severity_key_table"].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=True), encoding="utf-8")
    _plot_default_policy_vs_severity(summary, output_dir / "default_policy_coverage_by_severity.png", metric="coverage")
    _plot_default_policy_vs_severity(summary, output_dir / "default_policy_accuracy_by_severity.png", metric="accuracy")
    _plot_mid_policy_curves(results, output_dir / "mid_risk_coverage_curves.png")
    _write_run_notes(results, summary, output_dir)
    return summary


def _plot_subject_mid_curves(result: dict[str, Any], path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for condition_key, condition in result["conditions"].items():
        if condition_key != "clean" and condition["severity"] != "mid":
            continue
        curve = condition["policies"]["fusion"]["risk_coverage_curve"]
        plt.plot(curve["coverage"], curve["risk"], label=f"{_condition_label(condition)} fusion", linewidth=1.2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title(f"Subject {result['subject']:02d} mid-degradation fusion curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_default_policy_vs_severity(summary: dict[str, Any], path: Path, metric: str) -> None:
    severity_order = ["none", "low", "mid", "high"]
    artifacts = ["clean", "eog", "emg", "drop"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, artifact in zip(axes, ["eog", "emg", "drop"]):
        condition_by_severity = {"none": summary["conditions"].get("clean")}
        condition_by_severity.update(
            {
                c["severity"]: c
                for c in summary["conditions"].values()
                if c["artifact"] == artifact
            }
        )
        x = [s for s in severity_order if condition_by_severity.get(s) is not None]
        for policy in KEY_POLICIES:
            y = [condition_by_severity[s]["policies_mean"][policy][metric] for s in x]
            ax.plot(x, y, marker="o", linewidth=1.5, label=policy)
        ax.set_title(artifact.upper())
        ax.set_xlabel("Severity")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Default " + ("accepted accuracy" if metric == "accuracy" else metric))
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Default policy {metric} by artifact severity")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_mid_policy_curves(results: list[dict[str, Any]], path: Path) -> None:
    conditions = ["clean", "eog_mid", "emg_mid", "drop_mid"]
    fig, axes = plt.subplots(1, len(conditions), figsize=(15, 3.8), sharey=True)
    for ax, condition_key in zip(axes, conditions):
        available = [r["conditions"][condition_key] for r in results if condition_key in r["conditions"]]
        if not available:
            ax.set_visible(False)
            continue
        for policy in KEY_POLICIES:
            min_len = min(len(c["policies"][policy]["risk_coverage_curve"]["coverage"]) for c in available)
            coverage = np.mean(
                [c["policies"][policy]["risk_coverage_curve"]["coverage"][:min_len] for c in available],
                axis=0,
            )
            risk = np.mean(
                [c["policies"][policy]["risk_coverage_curve"]["risk"][:min_len] for c in available],
                axis=0,
            )
            ax.plot(coverage, risk, label=policy, linewidth=1.5)
        ax.set_title(condition_key.replace("_", "/"))
        ax.set_xlabel("Coverage")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Risk (1 - accuracy)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Pooled mean risk-coverage curves under mid degradation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _write_run_notes(results: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    key_rows = []
    for row in summary["mid_severity_key_table"]:
        key_rows.append(
            [
                row["condition"],
                f"{row['forced_accuracy']:.3f}",
                f"{row['sqi_mean']:.3f}",
                f"{row['softmax_coverage']:.3f}/{row['softmax_accepted_accuracy']:.3f}",
                f"{row['sqi_coverage']:.3f}/{row['sqi_accepted_accuracy']:.3f}",
                f"{row['fusion_coverage']:.3f}/{row['fusion_accepted_accuracy']:.3f}",
            ]
        )
    table = _markdown_table(
        ["Condition", "Forced acc", "Mean SQI", "Softmax cov/acc", "SQI cov/acc", "Fusion cov/acc"],
        key_rows,
    )
    artifact_names = sorted({p.name for p in output_dir.iterdir() if p.is_file()} | {"RUN_NOTES.md"})
    inventory = "\n".join(f"- `{name}`" for name in artifact_names)
    notes = f"""# BCIC IV 2a artifact degradation run

- Source commit: `{_git_head()}`
- Command: `python3 scripts/run_bcic2a_degradation.py --config configs/bcic2a_degradation.yaml`
- Config: `configs/bcic2a_degradation.yaml`
- Output directory: `{output_dir.as_posix()}/`
- Dataset: MOABB `BNCI2014_001` (BCIC IV 2a), `left_right_hand` paradigm
- Protocol: CSP-LDA and SQI are fit on clean train epochs only; EOG/EMG/drop degradation is injected into test epochs only.
- Completed subjects: {", ".join(str(r["subject"]) for r in results)}
- Failures/errors: none recorded by the runner.

## Mid-severity default operating points

Coverage/accuracy columns report default operating-point coverage and accepted accuracy.

{table}

## Artifact inventory

{inventory}
"""
    (output_dir / "RUN_NOTES.md").write_text(notes, encoding="utf-8")


def _subjects_from_args(args: argparse.Namespace, cfg: dict[str, Any]) -> list[int]:
    if args.subject:
        subjects = args.subject
    else:
        subjects = list(cfg.get("dataset", {}).get("subjects", list(range(1, 10))))
    if args.smoke_moabb:
        return [int(subjects[0])]
    return [int(s) for s in subjects]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BCIC IV 2a test-time artifact degradation experiments.")
    parser.add_argument("--config", default="configs/bcic2a_degradation.yaml")
    parser.add_argument("--subject", type=int, action="append", help="Subject to run; can be repeated. Overrides config list.")
    parser.add_argument("--artifact", choices=["eog", "emg", "drop"], action="append", help="Artifact type to run; can be repeated.")
    parser.add_argument("--severity", choices=["none", "low", "mid", "high"], action="append", help="Severity to run; can be repeated.")
    parser.add_argument("--smoke-moabb", action="store_true", help="Run only one configured subject to validate MOABB download/loading.")
    parser.add_argument("--output-dir", default=None, help="Override artifact output directory.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    output_dir = Path(args.output_dir or cfg.get("artifacts_dir", "artifacts/bcic2a_degradation"))
    subjects = _subjects_from_args(args, cfg)
    condition_specs = _condition_specs(cfg, selected_artifacts=args.artifact, selected_severities=args.severity)

    try:
        results = []
        for subject in subjects:
            print(f"Running BNCI2014_001 degradation CSP-LDA for subject {subject:02d}")
            result = _run_subject(subject, cfg, output_dir, condition_specs)
            results.append(result)
            clean = result["conditions"]["clean"]
            print(
                f"  clean_acc={clean['metrics']['accuracy']:.3f} "
                f"clean_sqi={clean['sqi']['test_mean']:.3f} conditions={len(result['conditions'])}"
            )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    summary = _write_summary(results, output_dir)
    print(f"Wrote per-subject JSON/PNG artifacts and summary under {output_dir}")
    print(json.dumps(_jsonable(summary["mid_severity_key_table"]), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
