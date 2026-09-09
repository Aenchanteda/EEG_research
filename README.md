# EEG_research
For code testing

## Quality-aware and Abstention-enabled Cross-dataset Motor Imagery EEG Decoding

This repository now scaffolds a reproducible Python experiment package,
`mi_eeg_qa`, for studying trust, calibration, signal quality, and abstention in
cross-dataset motor imagery EEG decoding. The package is framed around
quality-aware decision support rather than chasing state-of-the-art decoder
accuracy.

### What is included

- Synthetic motor imagery EEG generator for deterministic smoke tests.
- CSP-LDA baseline with `predict_proba`.
- PyTorch EEGNet and ShallowConvNet wrappers with `predict_proba`;
  EEGNet supports MC-dropout probability estimation through the wrapper.
  ShallowConvNet is kept as a lightweight stand-in for future FBCNet work.
- Signal quality index (SQI) features:
  peak-to-peak/RMS, high-frequency power proxy, and flat-channel fraction,
  mapped to `q in [0, 1]`.
- Abstention policies: forced accept, softmax max-probability, margin,
  temperature scaling, SQI-only, AND-combination, fusion, and threshold sweeps.
- Metrics: accuracy, expected calibration error (ECE, default 15 bins), Brier
  score, risk-coverage curve, and accuracy at coverage.
- Degradation helpers for EOG-like blinks, EMG-like high-frequency noise, and
  channel dropping.
- Optional MOABB/MNE loader entry points. The real within-subject script below
  targets BCIC IV 2a (`BNCI2014_001`) left-vs-right hand binary decoding.

### Install

Python 3.10+ is required.

```bash
python3 -m pip install -e .[dev]
```

For optional MOABB/MNE dataset loading:

```bash
python3 -m pip install -e .[dev,moabb]
```

### Run tests

```bash
pytest
```

### Run the smoke experiment

```bash
python3 scripts/run_smoke.py --config configs/smoke.yaml
```

The smoke run trains CSP-LDA on deterministic synthetic EEG, compares forced,
softmax, SQI-only, AND-combined, and fusion abstention policies, prints ECE and
Brier score, and writes:

- `artifacts/smoke_results.json`
- `artifacts/smoke_risk_coverage.png`

### Run BCIC IV 2a within-subject experiments

```bash
python3 scripts/run_within_subject.py --config configs/bcic2a.yaml
```

This command requires the optional `moabb` extra:

```bash
python3 -m pip install -e .[dev,moabb]
python3 scripts/run_within_subject.py --config configs/bcic2a.yaml
```

The first run downloads MOABB/MNE data. By default, the config runs all 9 BCIC
IV 2a subjects with the `left_right_hand` paradigm so CSP-LDA remains binary.
The script prefers session-based train/test splits from MOABB metadata
(BCIC IV 2a typically provides separate train/evaluation sessions). If session
metadata is unavailable, it falls back to a deterministic stratified holdout
using the configured seed.

Run one subject:

```bash
python3 scripts/run_within_subject.py --config configs/bcic2a.yaml --subject 1
```

Run a one-subject MOABB smoke/download check:

```bash
python3 scripts/run_within_subject.py --config configs/bcic2a.yaml --smoke-moabb
```

The BCIC run writes per-subject JSON and risk-coverage PNGs under
`artifacts/bcic2a/`, plus `summary.json` and
`summary_fusion_risk_coverage.png`. Saved policy IDs are locked to:
`forced`, `softmax`, `margin`, `sqi`, `combined_and`, and `fusion`; model ID
`csp_lda` maps to `CSPLDAClassifier`.

### Run BCIC IV 2a artifact degradation experiments

```bash
python3 scripts/run_bcic2a_degradation.py --config configs/bcic2a_degradation.yaml
```

This controlled experiment uses the same binary BCIC IV 2a left/right
within-subject setup as `run_within_subject.py`, fits CSP-LDA and SQI on clean
training epochs only, and injects EOG-like bursts, EMG/HF bursts, or channel
drops into test epochs. Outputs are written to
`artifacts/bcic2a_degradation/` as per-subject JSON files, pooled
`summary.json`, `RUN_NOTES.md`, and summary PNGs.

Run a smaller matrix first:

```bash
python3 scripts/run_bcic2a_degradation.py --config configs/bcic2a_degradation.yaml --severity none --severity mid
```

Run one subject or one artifact type:

```bash
python3 scripts/run_bcic2a_degradation.py --config configs/bcic2a_degradation.yaml --subject 1
python3 scripts/run_bcic2a_degradation.py --config configs/bcic2a_degradation.yaml --artifact eog
```
### Run BCIC IV 2b within-subject experiments

```bash
python3 -m pip install -e .[dev,moabb]
python3 scripts/run_within_subject.py --config configs/bcic2b.yaml
```

The BCIC IV 2b config uses MOABB `BNCI2014_004`, all 9 subjects, the same
binary `left_right_hand` paradigm, 8-30 Hz bandpass, and the locked abstention
policy IDs listed above. It writes artifacts under `artifacts/bcic2b/`.
BCIC IV 2b differs from 2a in channel montage: it has 3 bipolar EEG channels
(C3, Cz, C4 montage), so the config uses 2 CSP components rather than the
larger 2a setting. The runner still prefers MOABB session-based splits and
documents the deterministic stratified fallback if session metadata is absent.
