#!/usr/bin/env python3
"""Run cross-dataset CSP-LDA motor imagery experiments through MOABB."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from mi_eeg_qa.abstention import confidence_scores, fusion_score, margin_scores, threshold_sweep
from mi_eeg_qa.data.harmonization import (
    BCIC_2A_EEG_CHANNELS,
    common_channel_selection,
    harmonize_binary_labels,
)
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
WITHIN_BCIC2A_BASELINE = {
    "source": "PR #2 BCIC IV 2a within-subject CSP-LDA artifacts",
    "accuracy": 0.855,
    "expected_calibration_error": 0.049,
    "brier_score": 0.207,
}
PREVIOUS_PARTIAL_RESULTS = {
    "PhysionetMI->BNCI2014_001": {
        "source_subjects": 12,
        "target_subjects": 9,
        "accuracy": 0.622,
        "expected_calibration_error": 0.120,
        "brier_score": 0.487,
    },
    "BNCI2014_001->PhysionetMI": {
        "source_subjects": 9,
        "target_subjects": 6,
        "accuracy": 0.530,
        "expected_calibration_error": 0.137,
        "brier_score": 0.535,
    },
}


@dataclass
class DatasetBlock:
    name: str
    subjects: list[int]
    X: np.ndarray
    y: np.ndarray
    subject_index: np.ndarray
    labels: tuple[str, str]
    channels: tuple[str, ...]
    sfreq: float
    skipped: list[dict[str, Any]]


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


def _require_moabb():
    try:
        from moabb.datasets import BNCI2014_001, BNCI2014_004, PhysionetMI  # type: ignore
        from moabb.paradigms import LeftRightImagery  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MOABB/MNE loaders require optional dependencies. Install with `pip install -e .[moabb]`."
        ) from exc
    return {
        "BNCI2014_001": BNCI2014_001,
        "BNCI2014_004": BNCI2014_004,
        "PhysionetMI": PhysionetMI,
        "LeftRightImagery": LeftRightImagery,
    }


def _dataset_instance(name: str, subjects: list[int] | None = None):
    mods = _require_moabb()
    if name == "PhysionetMI":
        dataset = mods[name](imagined=True, executed=False, subjects=subjects)
        dataset.feet_runs = []
        return dataset
    if name in {"BNCI2014_001", "BNCI2014_004"}:
        return mods[name]()
    raise ValueError(f"Unsupported cross-dataset source/target {name!r}")


def _known_channels(name: str) -> tuple[str, ...]:
    mods = _require_moabb()
    if name == "BNCI2014_001":
        return BCIC_2A_EEG_CHANNELS
    if name == "BNCI2014_004":
        return ("C3", "Cz", "C4")
    if name == "PhysionetMI":
        sensors = mods[name].METADATA.acquisition.sensors
        return tuple(str(ch) for ch in sensors)
    raise ValueError(f"Unsupported dataset {name!r}")


def _subject_list(spec: dict[str, Any]) -> list[int]:
    name = spec["name"]
    subjects = spec.get("subjects", "all")
    exclude = {int(s) for s in spec.get("exclude_subjects", [])}
    if subjects == "all":
        ds = _dataset_instance(name)
        subjects = list(getattr(ds, "subject_list"))
    out = [int(s) for s in subjects if int(s) not in exclude]
    if not out:
        raise ValueError(f"No subjects configured for {name}")
    return out


def _configured_exclusions(spec: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = {int(k): str(v) for k, v in spec.get("exclude_reasons", {}).items()}
    out = []
    for subject in spec.get("exclude_subjects", []):
        subject = int(subject)
        out.append(
            {
                "subject": subject,
                "reason": reasons.get(
                    subject,
                    "configured exclusion; see dataset notes",
                ),
            }
        )
    return out


def _limit_subjects(subjects: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return subjects
    return subjects[: max(0, int(limit))]


def _load_dataset_block(
    spec: dict[str, Any],
    subjects: list[int],
    channels: tuple[str, ...],
    cfg: dict[str, Any],
) -> DatasetBlock:
    mods = _require_moabb()
    pp = cfg.get("preprocess", {})
    epoch = cfg.get("epoch", {})
    sfreq = float(cfg.get("sfreq", 160.0))
    paradigm = mods["LeftRightImagery"](
        fmin=float(pp.get("low", 8.0)),
        fmax=float(pp.get("high", 30.0)),
        tmin=float(epoch.get("tmin", 0.5)),
        tmax=float(epoch.get("tmax", 4.0)),
        channels=list(channels),
        resample=sfreq,
    )

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subject_index: list[np.ndarray] = []
    loaded_subjects: list[int] = []
    skipped: list[dict[str, Any]] = []
    aliases = spec.get("label_aliases", {})
    for subject in subjects:
        try:
            dataset = _dataset_instance(spec["name"], [subject])
            X, raw_y, _metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        except Exception as exc:  # pragma: no cover - depends on network/datasets
            skipped.append({"subject": int(subject), "reason": f"{type(exc).__name__}: {exc}"})
            print(f"Skipping {spec['name']} subject {subject}: {exc}", file=sys.stderr)
            continue
        y, labels = harmonize_binary_labels(raw_y, label_aliases=aliases)
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != len(channels):
            skipped.append(
                {
                    "subject": int(subject),
                    "reason": f"expected {len(channels)} channels, got {X.shape[1]}",
                }
            )
            continue
        xs.append(X)
        ys.append(y)
        subject_index.append(np.full(len(y), int(subject), dtype=int))
        loaded_subjects.append(int(subject))

    if not xs:
        raise RuntimeError(f"No {spec['name']} subjects loaded successfully")
    return DatasetBlock(
        name=str(spec["name"]),
        subjects=loaded_subjects,
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        subject_index=np.concatenate(subject_index, axis=0),
        labels=labels,
        channels=channels,
        sfreq=sfreq,
        skipped=skipped,
    )


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


def _evaluate_policies(y_true: np.ndarray, proba: np.ndarray, q: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
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


def _metrics(y_true: np.ndarray, proba: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, proba.argmax(axis=1)),
        "expected_calibration_error": expected_calibration_error(
            y_true,
            proba,
            n_bins=int(cfg.get("abstention", {}).get("ece_bins", 15)),
        ),
        "brier_score": brier_score(y_true, proba),
    }


def _preprocess(X: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    pp = cfg.get("preprocess", {})
    sfreq = float(cfg.get("sfreq", 160.0))
    X = bandpass_epochs(X, sfreq=sfreq, low=float(pp.get("low", 8.0)), high=float(pp.get("high", 30.0)))
    return standardize_epochs(X)


def _plot_pooled_risk(summary: dict[str, Any], path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for name in POLICIES:
        curve = summary["pooled"]["policies"][name]["risk_coverage_curve"]
        plt.plot(curve["coverage"], curve["risk"], label=name, linewidth=1.2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title(f"{summary['source']['name']} -> {summary['target']['name']} pooled risk-coverage")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_policy_table(summary: dict[str, Any], path: Path) -> None:
    policies = list(POLICIES)
    coverage = [summary["pooled"]["policies"][p]["default"]["coverage"] for p in policies]
    acc = [summary["pooled"]["policies"][p]["default"]["accuracy"] for p in policies]
    x = np.arange(len(policies))
    width = 0.38
    plt.figure(figsize=(8.5, 4.5))
    plt.bar(x - width / 2, coverage, width, label="Coverage")
    plt.bar(x + width / 2, acc, width, label="Accepted accuracy")
    plt.xticks(x, policies, rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Default abstention policy operating points")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _write_run_notes(summary: dict[str, Any], output_dir: Path) -> None:
    pooled = summary["pooled"]["metrics"]
    lines = [
        "# Cross-dataset PhysioNet/BCIC RUN_NOTES",
        "",
        f"Config: `{summary['config_path']}`",
        f"Direction: {summary['source']['name']} -> {summary['target']['name']}",
        "",
        "## Protocol",
        "",
        "- Binary CSP-LDA only; no EEGNet, FBCNet, or domain-adversarial model is used.",
        "- Source subjects are pooled into one labeled training set.",
        "- Target labels are used only for evaluation, never for SQI fitting, threshold fitting, or model training.",
        "- SQI is fit on the source training epochs and transformed on target test epochs.",
        "- Locked policy IDs: forced, softmax, margin, sqi, combined_and, fusion.",
        "",
        "## Harmonization",
        "",
        "- Labels: left_hand vs right_hand only.",
        "- PhysioNet: MOABB PhysionetMI with imagined=True and executed=False, selecting imagery left/right fist runs 4, 8, and 12 through the dataset wrapper.",
        "- Channels: deterministic intersection in BCIC IV 2a 22-channel order.",
        f"- Selected channels ({len(summary['channels'])}): {', '.join(summary['channels'])}.",
        f"- Epoch window: {summary['epoch']['tmin']} to {summary['epoch']['tmax']} s.",
        f"- Common sampling frequency: {summary['sfreq']} Hz.",
        "",
        "## Subject counts",
        "",
        f"- Source configured subjects: {summary['source']['configured_subject_count']}",
        f"- Source run limit: {summary['source']['run_subject_limit'] if summary['source']['run_subject_limit'] is not None else 'none'}",
        f"- Source loaded subjects: {len(summary['source']['subjects'])} ({summary['source']['subjects']})",
        f"- Source configured exclusions: {summary['source'].get('configured_exclusions', [])}",
        f"- Target configured subjects: {summary['target']['configured_subject_count']}",
        f"- Target run limit: {summary['target']['run_subject_limit'] if summary['target']['run_subject_limit'] is not None else 'none'}",
        f"- Target loaded subjects: {len(summary['target']['subjects'])} ({summary['target']['subjects']})",
        f"- Target configured exclusions: {summary['target'].get('configured_exclusions', [])}",
        "- PhysioNet subject 88 is excluded by config for full-dataset loads because MOABB documents it at 128 Hz rather than 160 Hz.",
        "",
        "## Pooled target metrics",
        "",
        f"- Accuracy: {pooled['accuracy']:.3f}",
        f"- ECE: {pooled['expected_calibration_error']:.3f}",
        f"- Brier: {pooled['brier_score']:.3f}",
        "",
        "## Default policy table",
        "",
        "| Policy | Coverage | Accepted accuracy | Risk |",
        "| --- | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        row = summary["pooled"]["policies"][policy]["default"]
        lines.append(f"| {policy} | {row['coverage']:.3f} | {row['accuracy']:.3f} | {row['risk']:.3f} |")
    lines.extend(
        [
            "",
            "## BCIC IV 2a within-subject reference",
            "",
            f"- Source: {WITHIN_BCIC2A_BASELINE['source']}",
            f"- Accuracy: {WITHIN_BCIC2A_BASELINE['accuracy']:.3f}",
            f"- ECE: {WITHIN_BCIC2A_BASELINE['expected_calibration_error']:.3f}",
            f"- Brier: {WITHIN_BCIC2A_BASELINE['brier_score']:.3f}",
            "- Cross-dataset accuracy is expected to drop under montage, subject-population, and collection-protocol shift.",
            "",
            "## Previous partial-run comparison",
            "",
            "| Run | Source subjects | Target subjects | Accuracy | ECE | Brier |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    direction_key = f"{summary['source']['name']}->{summary['target']['name']}"
    previous = summary.get("previous_partial_reference") or PREVIOUS_PARTIAL_RESULTS.get(direction_key)
    if previous:
        lines.append(
            f"| Previous partial | {previous['source_subjects']} | {previous['target_subjects']} | "
            f"{previous['accuracy']:.3f} | {previous['expected_calibration_error']:.3f} | {previous['brier_score']:.3f} |"
        )
    lines.append(
        f"| Expanded current | {len(summary['source']['subjects'])} | {len(summary['target']['subjects'])} | "
        f"{pooled['accuracy']:.3f} | {pooled['expected_calibration_error']:.3f} | {pooled['brier_score']:.3f} |"
    )
    lines.extend(
        [
            "",
            "## Skips",
            "",
        ]
    )
    for side in ("source", "target"):
        skipped = summary[side]["skipped"]
        if not skipped:
            lines.append(f"- {side}: none")
        else:
            for item in skipped:
                lines.append(f"- {side} subject {item['subject']}: {item['reason']}")
    (output_dir / "RUN_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source_spec = cfg["source"]
    target_spec = cfg["target"]
    configured_source_subjects = _subject_list(source_spec)
    configured_target_subjects = _subject_list(target_spec)
    source_subjects = _limit_subjects(configured_source_subjects, args.max_source_subjects)
    target_subjects = _limit_subjects(configured_target_subjects, args.max_target_subjects)

    source_channels = _known_channels(source_spec["name"])
    target_channels = _known_channels(target_spec["name"])
    selection = common_channel_selection(source_channels, target_channels, preferred_order=BCIC_2A_EEG_CHANNELS)
    channels = selection.channels

    source = _load_dataset_block(source_spec, source_subjects, channels, cfg)
    target = _load_dataset_block(target_spec, target_subjects, channels, cfg)

    X_source = _preprocess(source.X, cfg)
    X_target = _preprocess(target.X, cfg)

    sqi_cfg = cfg.get("sqi", {})
    sqi_scorer = fit_sqi(X_source, sfreq=source.sfreq, method=sqi_cfg.get("method", "rank"))
    q_source = transform_sqi(sqi_scorer, X_source)
    q_target = transform_sqi(sqi_scorer, X_target)

    model_cfg = cfg.get("model", {})
    if model_cfg.get("name", "csp_lda") != "csp_lda":
        raise ValueError("Cross-dataset script currently supports model.name: csp_lda")
    model = CSPLDAClassifier(n_components=int(model_cfg.get("csp_components", 8)))
    model.fit(X_source, source.y)
    proba_target = model.predict_proba(X_target)

    per_subject = []
    output_dir = Path(args.output_dir or cfg.get("artifacts_dir", "artifacts/cross_physionet_bcic"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for subject in target.subjects:
        mask = target.subject_index == subject
        subject_result = {
            "subject": int(subject),
            "n_test": int(mask.sum()),
            "metrics": _metrics(target.y[mask], proba_target[mask], cfg),
            "sqi": {"test_mean": float(np.mean(q_target[mask]))},
            "policies": _evaluate_policies(target.y[mask], proba_target[mask], q_target[mask], cfg),
        }
        per_subject.append(subject_result)
        (output_dir / f"target_subject_{subject:03d}.json").write_text(
            json.dumps(_jsonable(subject_result), indent=2, allow_nan=True),
            encoding="utf-8",
        )

    pooled = {
        "n_test": int(len(target.y)),
        "metrics": _metrics(target.y, proba_target, cfg),
        "sqi": {
            "source_train_mean": float(np.mean(q_source)),
            "target_test_mean": float(np.mean(q_target)),
            "method": sqi_cfg.get("method", "rank"),
        },
        "policies": _evaluate_policies(target.y, proba_target, q_target, cfg),
    }
    summary = {
        "config_path": str(args.config),
        "source": {
            "name": source.name,
            "subjects": source.subjects,
            "configured_subject_count": len(configured_source_subjects),
            "run_subject_limit": args.max_source_subjects,
            "n_train": int(len(source.y)),
            "skipped": source.skipped,
        },
        "target": {
            "name": target.name,
            "subjects": target.subjects,
            "configured_subject_count": len(configured_target_subjects),
            "run_subject_limit": args.max_target_subjects,
            "n_test": int(len(target.y)),
            "skipped": target.skipped,
        },
        "labels": source.labels,
        "channels": channels,
        "sfreq": source.sfreq,
        "epoch": cfg.get("epoch", {}),
        "model": model_cfg,
        "within_subject_baseline_bcic2a": WITHIN_BCIC2A_BASELINE,
        "previous_partial_reference": PREVIOUS_PARTIAL_RESULTS.get(f"{source.name}->{target.name}"),
        "pooled": pooled,
        "per_target_subject": per_subject,
    }
    summary["source"]["configured_exclusions"] = _configured_exclusions(source_spec)
    summary["target"]["configured_exclusions"] = _configured_exclusions(target_spec)
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=True), encoding="utf-8")
    _write_run_notes(summary, output_dir)
    _plot_pooled_risk(summary, output_dir / "pooled_risk_coverage.png")
    _plot_policy_table(summary, output_dir / "pooled_policy_operating_points.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-dataset PhysioNet/BCIC CSP-LDA experiments.")
    parser.add_argument("--config", default="configs/cross_physionet_to_bcic2a.yaml")
    parser.add_argument("--output-dir", default=None, help="Override artifact output directory.")
    parser.add_argument("--max-source-subjects", type=int, default=None, help="Limit source subjects for partial runs.")
    parser.add_argument("--max-target-subjects", type=int, default=None, help="Limit target subjects for partial runs.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    try:
        summary = _run(cfg, args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    m = summary["pooled"]["metrics"]
    print(
        f"{summary['source']['name']} -> {summary['target']['name']}: "
        f"accuracy={m['accuracy']:.3f} ece={m['expected_calibration_error']:.3f} "
        f"brier={m['brier_score']:.3f} n_test={summary['target']['n_test']}"
    )
    print(f"Wrote cross-dataset artifacts under {Path(args.output_dir or cfg.get('artifacts_dir'))}")


if __name__ == "__main__":
    main()
