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
- PyTorch EEGNet and ShallowConvNet/FBCNet-like wrappers with `predict_proba`;
  EEGNet supports MC-dropout probability estimation through the wrapper.
- Signal quality index (SQI) features:
  peak-to-peak/RMS, high-frequency power proxy, and flat-channel fraction,
  mapped to `q in [0, 1]`.
- Abstention policies: forced accept, softmax max-probability, margin,
  temperature scaling, SQI-only, AND-combination, fusion, and threshold sweeps.
- Metrics: accuracy, expected calibration error (ECE, default 15 bins), Brier
  score, risk-coverage curve, and accuracy at coverage.
- Degradation helpers for EOG-like blinks, EMG-like high-frequency noise, and
  channel dropping.
- Optional MOABB/MNE loader entry points for BCIC IV 2a/2b and PhysioNet MI.

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

### Optional within-subject loader check

```bash
python3 scripts/run_within_subject.py --config configs/bcic2a.yaml
```

This command requires the optional `moabb` extra. It currently verifies loading
and reports the dataset shape; full cross-dataset protocol orchestration is left
as the next extension point.
