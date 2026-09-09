# BCIC IV 2a within-subject EEGNet abstention run

- Date: 2026-09-09 UTC
- Command: `python3 scripts/run_within_subject.py --config configs/bcic2a_eegnet_mid_emg.yaml`
- Config: `configs/bcic2a_eegnet_mid_emg.yaml`
- Output directory: `artifacts/bcic2a_eegnet_mid_emg/`
- Dataset: MOABB `BNCI2014_001` (BCIC IV 2a), `left_right_hand` paradigm
- Model: `eegnet`
- Training budget: seed 7, epochs 12, batch size 32, lr 0.001
- Preprocess: 8-30 Hz bandpass, per-epoch/channel standardization
- SQI: fitted on train split only and transformed on test/evaluation trials
- Completed subjects: 1, 2, 3, 4, 5, 6, 7, 8, 9
- CSP-LDA reference: Acc 0.855, ECE 0.049, Brier 0.207
- Test-time degradation: mid_emg emg after clean train preprocessing (strength 0.6, probability 0.3)

## Subject-mean metrics

| Metric | Mean | CSP-LDA reference |
| --- | ---: | ---: |
| Accuracy | 0.856 | 0.855 |
| ECE | 0.180 | 0.049 |
| Brier | 0.294 | 0.207 |

## Default policy means

| Policy | Coverage | Accepted accuracy | Risk |
| --- | ---: | ---: | ---: |
| forced | 1.000 | 0.856 | 0.144 |
| softmax | 0.552 | 0.949 | 0.051 |
| margin | 0.552 | 0.949 | 0.051 |
| sqi | 0.248 | 0.885 | 0.115 |
| combined_and | 0.144 | 0.983 | 0.017 |
| fusion | 0.485 | 0.921 | 0.079 |

## Per-subject compact metrics

| Subject | Accuracy | ECE | Brier | Forced cov | Softmax cov/acc | Margin cov/acc | SQI cov/acc | Combined cov/acc | Fusion cov/acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.868 | 0.255 | 0.331 | 1.000 | 0.562/0.975 | 0.562/0.975 | 0.236/0.882 | 0.132/1.000 | 0.368/0.943 |
| 2 | 0.764 | 0.204 | 0.426 | 1.000 | 0.201/0.931 | 0.201/0.931 | 0.194/0.857 | 0.056/1.000 | 0.222/0.875 |
| 3 | 0.931 | 0.161 | 0.154 | 1.000 | 0.861/0.976 | 0.861/0.976 | 0.292/0.976 | 0.257/1.000 | 0.715/0.981 |
| 4 | 0.764 | 0.112 | 0.365 | 1.000 | 0.694/0.810 | 0.694/0.810 | 0.250/0.861 | 0.153/0.909 | 0.528/0.895 |
| 5 | 0.819 | 0.259 | 0.416 | 1.000 | 0.181/0.923 | 0.181/0.923 | 0.229/0.818 | 0.056/1.000 | 0.215/0.871 |
| 6 | 0.819 | 0.243 | 0.391 | 1.000 | 0.299/1.000 | 0.299/1.000 | 0.229/0.879 | 0.076/1.000 | 0.347/0.900 |
| 7 | 0.840 | 0.262 | 0.380 | 1.000 | 0.299/1.000 | 0.299/1.000 | 0.271/0.821 | 0.090/1.000 | 0.306/0.886 |
| 8 | 0.938 | 0.054 | 0.096 | 1.000 | 0.931/0.955 | 0.931/0.955 | 0.271/0.923 | 0.243/0.971 | 0.806/0.974 |
| 9 | 0.965 | 0.074 | 0.088 | 1.000 | 0.938/0.970 | 0.938/0.970 | 0.257/0.946 | 0.236/0.971 | 0.854/0.967 |

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
