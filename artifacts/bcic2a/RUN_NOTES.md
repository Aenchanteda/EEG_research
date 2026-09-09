# BCIC IV 2a within-subject CSP-LDA abstention run

- Date: 2026-09-09 UTC
- Source commit: `d2a7635162a3d0747da56f778e141614f3a3da62`
- Command: `python3 scripts/run_within_subject.py --config configs/bcic2a.yaml`
- Config: `configs/bcic2a.yaml`
- Output directory: `artifacts/bcic2a/`
- Dataset: MOABB `BNCI2014_001` (BCIC IV 2a), `left_right_hand` paradigm
- Completed subjects: 1, 2, 3, 4, 5, 6, 7, 8, 9
- Failures/errors: none. MOABB emitted a benign first-run MNE config warning and an initial README manifest fallback while downloading; all data downloads and subjects completed successfully.

## Pooled summary

| Metric | Mean |
| --- | ---: |
| Accuracy | 0.855 |
| ECE | 0.049 |
| Brier | 0.207 |

## Default policy means

| Policy | Coverage | Accepted accuracy | Risk |
| --- | ---: | ---: | ---: |
| forced | 1.000 | 0.855 | 0.145 |
| softmax | 0.889 | 0.878 | 0.122 |
| margin | 0.889 | 0.878 | 0.122 |
| sqi | 0.333 | 0.846 | 0.154 |
| combined_and | 0.302 | 0.866 | 0.134 |
| fusion | 0.821 | 0.889 | 0.111 |

## Per-subject compact metrics

| Subject | Accuracy | ECE | Brier | Forced cov | Softmax cov/acc | Margin cov/acc | SQI cov/acc | Combined cov/acc | Fusion cov/acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.910 | 0.036 | 0.150 | 1.000 | 0.944/0.926 | 0.944/0.926 | 0.340/0.939 | 0.326/0.957 | 0.868/0.944 |
| 2 | 0.715 | 0.064 | 0.400 | 1.000 | 0.729/0.752 | 0.729/0.752 | 0.236/0.618 | 0.160/0.739 | 0.597/0.709 |
| 3 | 0.958 | 0.028 | 0.065 | 1.000 | 0.979/0.965 | 0.979/0.965 | 0.389/0.946 | 0.375/0.963 | 0.986/0.958 |
| 4 | 0.812 | 0.060 | 0.254 | 1.000 | 0.889/0.852 | 0.889/0.852 | 0.340/0.837 | 0.312/0.844 | 0.799/0.870 |
| 5 | 0.688 | 0.069 | 0.393 | 1.000 | 0.715/0.738 | 0.715/0.738 | 0.299/0.698 | 0.229/0.727 | 0.597/0.779 |
| 6 | 0.826 | 0.039 | 0.247 | 1.000 | 0.896/0.853 | 0.896/0.853 | 0.326/0.830 | 0.292/0.810 | 0.812/0.889 |
| 7 | 0.882 | 0.058 | 0.169 | 1.000 | 0.917/0.902 | 0.917/0.902 | 0.354/0.824 | 0.333/0.833 | 0.889/0.922 |
| 8 | 0.972 | 0.033 | 0.046 | 1.000 | 0.972/0.971 | 0.972/0.971 | 0.347/1.000 | 0.347/1.000 | 0.924/1.000 |
| 9 | 0.931 | 0.057 | 0.135 | 1.000 | 0.958/0.942 | 0.958/0.942 | 0.361/0.923 | 0.347/0.920 | 0.917/0.932 |

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
