# BCIC IV 2b within-subject CSP-LDA abstention run

- Command: `python3 scripts/run_within_subject.py --config configs/bcic2b.yaml`
- Config: `configs/bcic2b.yaml`
- Output directory: `artifacts/bcic2b/`
- Dataset: MOABB `BNCI2014_004`, `left_right_hand` paradigm
- Subjects completed: 1, 2, 3, 4, 5, 6, 7, 8, 9
- Bandpass: 8.0-30.0 Hz
- Model: `csp_lda` with 2 CSP components
- Channel montage note: BCIC IV 2b uses 3 bipolar EEG channels (C3, Cz, C4 montage) rather than the 22-channel BCIC IV 2a montage.
- SQI: fit on train split only, then transformed train/test with method `rank`
- Split methods observed: session
- Failures/errors: none.

## Pooled held-out metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.628 |
| ECE | 0.026 |
| Brier | 0.447 |
| Trials | 2840 |

## Subject-mean metrics

| Metric | Mean |
| --- | ---: |
| Accuracy | 0.626 |
| ECE | 0.042 |
| Brier | 0.448 |

## Default policy means

| Policy | Coverage | Accepted accuracy | Risk |
| --- | ---: | ---: | ---: |
| forced | 1.000 | 0.626 | 0.374 |
| softmax | 0.405 | 0.658 | 0.342 |
| margin | 0.405 | 0.658 | 0.342 |
| sqi | 0.315 | 0.619 | 0.381 |
| combined_and | 0.126 | 0.784 | 0.216 |
| fusion | 0.429 | 0.642 | 0.358 |

## Per-subject compact metrics

| Subject | Accuracy | ECE | Brier | Forced cov | Softmax cov/acc | Margin cov/acc | SQI cov/acc | Combined cov/acc | Fusion cov/acc | Split |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.613 | 0.024 | 0.465 | 1.000 | 0.397/0.677 | 0.397/0.677 | 0.331/0.594 | 0.122/0.564 | 0.416/0.632 | session |
| 2 | 0.532 | 0.025 | 0.497 | 1.000 | 0.004/0.000 | 0.004/0.000 | 0.336/0.479 | 0.000/nan | 0.250/0.414 | session |
| 3 | 0.537 | 0.025 | 0.490 | 1.000 | 0.084/0.741 | 0.084/0.741 | 0.325/0.567 | 0.028/1.000 | 0.328/0.581 | session |
| 4 | 0.625 | 0.077 | 0.447 | 1.000 | 0.450/0.764 | 0.450/0.764 | 0.297/0.568 | 0.144/0.739 | 0.412/0.644 | session |
| 5 | 0.725 | 0.068 | 0.377 | 1.000 | 0.634/0.818 | 0.634/0.818 | 0.206/0.742 | 0.125/0.925 | 0.472/0.801 | session |
| 6 | 0.606 | 0.022 | 0.473 | 1.000 | 0.475/0.651 | 0.475/0.651 | 0.381/0.639 | 0.181/0.759 | 0.516/0.636 | session |
| 7 | 0.719 | 0.041 | 0.382 | 1.000 | 0.772/0.769 | 0.772/0.769 | 0.344/0.700 | 0.269/0.767 | 0.656/0.748 | session |
| 8 | 0.641 | 0.052 | 0.446 | 1.000 | 0.406/0.769 | 0.406/0.769 | 0.350/0.652 | 0.138/0.818 | 0.431/0.659 | session |
| 9 | 0.641 | 0.043 | 0.452 | 1.000 | 0.422/0.733 | 0.422/0.733 | 0.263/0.631 | 0.125/0.700 | 0.381/0.664 | session |

## Artifact inventory

- `summary.json`
- `summary_fusion_risk_coverage.png`
- `subject_01.json`, `subject_01_risk_coverage.png`
- `subject_02.json`, `subject_02_risk_coverage.png`
- `subject_03.json`, `subject_03_risk_coverage.png`
- `subject_04.json`, `subject_04_risk_coverage.png`
- `subject_05.json`, `subject_05_risk_coverage.png`
- `subject_06.json`, `subject_06_risk_coverage.png`
- `subject_07.json`, `subject_07_risk_coverage.png`
- `subject_08.json`, `subject_08_risk_coverage.png`
- `subject_09.json`, `subject_09_risk_coverage.png`
