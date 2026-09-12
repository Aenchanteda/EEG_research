# Evaluating Signal Quality and Confidence for Selective Motor Imagery EEG Decoding

Companion code for the Sensors manuscript of the same title (Nan et al.).

> **Authoritative analysis package:** use the accompanying **`reproducibility_package`** (corrected runners, configs, regenerated predictions, and table/figure scripts).
> The historical GitHub snapshot at this repository is **not** the corrected analysis used for the manuscript tables.

## What this manuscript evaluates

Independent-session MI-EEG abstention with a training-fitted signal quality index (SQI), maximum posterior confidence, and fixed equal-weight fusion, compared at **matched retained-trial counts** (nominal 80% / 90% coverage).

Main evidence:
- CSP-LDA on BCI Competition IV 2a and 2b (nine subjects each)
- Fixed-budget EEGNet-style check on 2a
- Synthetic post-preprocessing perturbations (EOG-like / EMG-like / channel drop)
- Separately audited source-only cross-corpus transfer (PhysioNet ↔ BCIC 2a)

**Not main evidence** (scaffold only): MC-dropout inference, temperature scaling, ShallowConvNet, AND-combination thresholds. Configuration threshold knobs (e.g. softmax/SQI/fusion cutoffs) define separate diagnostics; they do **not** define the matched-coverage table columns.

## Data availability

### Raw EEG (do not rehost)
- **BCI Competition IV** datasets 2a / 2b — public; load via **MOABB**
- **PhysioNet** EEG Motor Movement/Imagery — public; load via **MOABB** / PhysioBank

This project does **not** redistribute raw EEG recordings.

### Derived artifacts
- Regenerated within-subject and perturbation predictions (e.g. **126** `.npz` arrays) live in the **reproducibility package** / local `artifacts/` trees used by `analyze_corrected.py`
- Archived transfer risk curves are audited numerically (not a fresh raw-data retrain in the manuscript analysis)

## Reproduce manuscript numbers

Follow the README inside the accompanying **reproducibility_package**:
1. `analyze_corrected.py` — regenerate numerical tables / supplementary curves from saved predictions
2. `score_diagnostics.py` — score diagnostics
3. `redesign_figures.py` — main figures (and optional graphical abstract)

Original source commit recorded in the package provenance: `fc34d6c792343e79cc7a4961eaf5a4ade5b062db` (superseded for within-subject evidence after session-split correction).

## Repository layout (corrected tree)

```text
mi_eeg_qa/          # package code
configs/            # experiment configs
scripts/            # runners
artifacts/          # derived outputs (predictions, summaries)
tests/              # unit / smoke tests
```

## Install (development)

Python 3.10+ recommended:

```bash
python3 -m pip install -e ".[dev]"
# optional real-data loaders:
python3 -m pip install -e ".[dev,moabb]"
pytest
```

## Citation

Please cite the Sensors manuscript (title above) and point readers to the **reproducibility_package** for exact prediction-level evidence. A Zenodo DOI for the package is pending author release.

---

### Historical scaffold notes

The sections below retain broader repository capabilities (smoke synthetic EEG, unused abstention options, etc.). They describe engineering scaffolding and **do not expand** the manuscript evidence scope.

## Scaffold package capabilities

- Synthetic motor imagery EEG generator for deterministic smoke tests.
- CSP-LDA baseline with `predict_proba`.
- PyTorch EEGNet and ShallowConvNet wrappers with `predict_proba`; EEGNet supports MC-dropout probability estimation through the wrapper.
- Signal quality index (SQI) features: peak-to-peak/RMS, high-frequency power proxy, and flat-channel fraction, mapped to `q in [0, 1]`.
- Abstention policies used for engineering diagnostics: forced accept, softmax max-probability, margin, temperature scaling, SQI-only, AND-combination, fusion, and threshold sweeps.
- Metrics helpers: accuracy, expected calibration error (ECE, default 15 bins), Brier score, risk-coverage curve, and accuracy at coverage.
- Degradation helpers for EOG-like blinks, EMG-like high-frequency noise, and channel dropping.
- Optional MOABB/MNE loader entry points for public datasets.

## Run scaffold tests and smoke checks

```bash
pytest
python3 scripts/run_smoke.py --config configs/smoke.yaml
```

The smoke run trains CSP-LDA on deterministic synthetic EEG, compares several diagnostic abstention policies, prints ECE and Brier score, and writes scaffold outputs such as `artifacts/smoke_results.json` and `artifacts/smoke_risk_coverage.png`.

## Historical runner examples

These commands document repository runner interfaces. Use the accompanying `reproducibility_package`, not this historical snapshot alone, for manuscript table reproduction.

```bash
# BCIC IV 2a CSP-LDA within-subject scaffold run
python3 scripts/run_within_subject.py --config configs/bcic2a.yaml

# One-subject MOABB smoke/download check
python3 scripts/run_within_subject.py --config configs/bcic2a.yaml --smoke-moabb

# BCIC IV 2a EEGNet-style scaffold run
python3 scripts/run_within_subject.py --config configs/bcic2a_eegnet.yaml

# BCIC IV 2a post-preprocessing degradation scaffold run
python3 scripts/run_bcic2a_degradation.py --config configs/bcic2a_degradation.yaml

# Smaller degradation matrix
python3 scripts/run_bcic2a_degradation.py --config configs/bcic2a_degradation.yaml --severity none --severity mid

# BCIC IV 2b CSP-LDA within-subject scaffold run
python3 scripts/run_within_subject.py --config configs/bcic2b.yaml
```

The MOABB-based runners download public dataset files through MOABB/MNE on first use. This repository does not host those raw recordings.
