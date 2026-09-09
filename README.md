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

### Run BCIC IV 2a EEGNet within-subject experiments

```bash
python3 scripts/run_within_subject.py --config configs/bcic2a_eegnet.yaml
```

This uses the same BCIC IV 2a binary left/right, 8-30 Hz preprocessing,
session split, train-only SQI fitting, and abstention policies as the CSP-LDA
run, but sets model ID `eegnet` to `EEGNetClassifier`. The documented training
budget is seed 7, 12 epochs, batch size 32, AdamW lr 0.001, weight decay
0.0001, and CPU execution. Outputs are written under
`artifacts/bcic2a_eegnet/` with `summary.json`, `RUN_NOTES.md`, per-subject
JSON/PNGs, and summary plots.

The optional story-scoped mid-EMG test-time degradation run trains the same
EEGNet models clean, injects EMG noise into test trials only, and writes a
separate artifact tree:

```bash
python3 scripts/run_within_subject.py --config configs/bcic2a_eegnet_mid_emg.yaml
```

### Run PhysioNet <-> BCIC IV 2a cross-dataset experiments

```bash
python3 -m pip install -e .[dev,moabb]
python3 scripts/run_cross_dataset.py --config configs/cross_physionet_to_bcic2a.yaml
python3 scripts/run_cross_dataset.py --config configs/cross_bcic2a_to_physionet.yaml
```

These runs keep the binary CSP-LDA setup and locked abstention policy IDs used
by the within-subject runner. The source dataset is pooled into one labeled
training set; the target dataset is used only for evaluation. SQI is fit on
source training epochs only and transformed on target epochs.

Harmonization is explicit in the config and `RUN_NOTES.md`: labels are
left-vs-right hand only, PhysioNet uses MOABB `PhysionetMI(imagined=True,
executed=False)` for imagery left/right fist runs 4/8/12, channels are selected
as the BCIC IV 2a 22-channel intersection, data are resampled to 160 Hz, and
the epoch window is 0.5-3.0 s. PhysioNet subject 88 is excluded for full
multi-subject loads because MOABB documents its 128 Hz sampling rate mismatch.

If full PhysioNet downloads are too slow, run a partial source or target check:

```bash
python3 scripts/run_cross_dataset.py --config configs/cross_physionet_to_bcic2a.yaml --max-source-subjects 20
python3 scripts/run_cross_dataset.py --config configs/cross_bcic2a_to_physionet.yaml --max-target-subjects 20
```

Artifacts are written under `artifacts/cross_physionet_bcic/`, including
`summary.json`, `RUN_NOTES.md`, pooled risk-coverage PNGs, default policy
operating-point PNGs, and per-target-subject JSON files. The summary includes
the BCIC IV 2a within-subject reference from PR #2 so the expected accuracy
drop under dataset shift can be reported without claiming FBCNet or other
out-of-scope models.
