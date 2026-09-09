# BCIC IV 2a within-subject EEGNet abstention run

- Date: 2026-09-09 UTC
- Command: `python3 scripts/run_within_subject.py --config configs/bcic2a_eegnet.yaml`
- Config: `configs/bcic2a_eegnet.yaml`
- Output directory: `artifacts/bcic2a_eegnet/`
- Dataset: MOABB `BNCI2014_001` (BCIC IV 2a), `left_right_hand` paradigm
- Model: `eegnet`
- Training budget: seed 7, epochs 12, batch size 32, lr 0.001
- Preprocess: 8-30 Hz bandpass, per-epoch/channel standardization
- SQI: fitted on train split only and transformed on test/evaluation trials
- Completed subjects: 1, 2, 3, 4, 5, 6, 7, 8, 9
- CSP-LDA reference: Acc 0.855, ECE 0.049, Brier 0.207
- Test-time degradation: none (clean primary run)

## Subject-mean metrics

| Metric | Mean | CSP-LDA reference |
| --- | ---: | ---: |
| Accuracy | 0.858 | 0.855 |
| ECE | 0.183 | 0.049 |
| Brier | 0.290 | 0.207 |

## Default policy means

| Policy | Coverage | Accepted accuracy | Risk |
| --- | ---: | ---: | ---: |
| forced | 1.000 | 0.858 | 0.142 |
| softmax | 0.538 | 0.961 | 0.039 |
| margin | 0.538 | 0.961 | 0.039 |
| sqi | 0.333 | 0.891 | 0.109 |
| combined_and | 0.194 | 0.978 | 0.022 |
| fusion | 0.566 | 0.921 | 0.079 |

## Per-subject compact metrics

| Subject | Accuracy | ECE | Brier | Forced cov | Softmax cov/acc | Margin cov/acc | SQI cov/acc | Combined cov/acc | Fusion cov/acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.889 | 0.282 | 0.335 | 1.000 | 0.479/0.986 | 0.479/0.986 | 0.340/0.918 | 0.167/1.000 | 0.479/0.957 |
| 2 | 0.743 | 0.184 | 0.426 | 1.000 | 0.201/0.931 | 0.201/0.931 | 0.236/0.882 | 0.056/1.000 | 0.271/0.897 |
| 3 | 0.944 | 0.155 | 0.145 | 1.000 | 0.861/0.992 | 0.861/0.992 | 0.389/0.964 | 0.340/1.000 | 0.826/0.975 |
| 4 | 0.799 | 0.174 | 0.338 | 1.000 | 0.625/0.911 | 0.625/0.911 | 0.340/0.816 | 0.215/0.903 | 0.604/0.862 |
| 5 | 0.792 | 0.240 | 0.417 | 1.000 | 0.160/0.913 | 0.160/0.913 | 0.299/0.837 | 0.069/1.000 | 0.292/0.881 |
| 6 | 0.819 | 0.242 | 0.391 | 1.000 | 0.319/0.978 | 0.319/0.978 | 0.326/0.894 | 0.118/0.941 | 0.444/0.906 |
| 7 | 0.833 | 0.254 | 0.378 | 1.000 | 0.333/1.000 | 0.333/1.000 | 0.354/0.804 | 0.125/1.000 | 0.396/0.877 |
| 8 | 0.944 | 0.055 | 0.091 | 1.000 | 0.924/0.970 | 0.924/0.970 | 0.347/0.940 | 0.312/0.978 | 0.875/0.968 |
| 9 | 0.958 | 0.063 | 0.087 | 1.000 | 0.938/0.970 | 0.938/0.970 | 0.361/0.962 | 0.340/0.980 | 0.903/0.969 |

## Artifact inventory

- `summary.json`
- `summary_fusion_risk_coverage.png`
- `summary_default_policy_operating_points.png`
- `subject_01.json`, `subject_01_risk_coverage.png`
- `subject_02.json`, `subject_02_risk_coverage.png`
- `subject_03.json`, `subject_03_risk_coverage.png`
- `subject_04.json`, `subject_04_risk_coverage.png`
- `subject_05.json`, `subject_05_risk_coverage.png`
- `subject_06.json`, `subject_06_risk_coverage.png`
- `subject_07.json`, `subject_07_risk_coverage.png`
- `subject_08.json`, `subject_08_risk_coverage.png`
- `subject_09.json`, `subject_09_risk_coverage.png`
