# BCIC IV 2a artifact degradation run

- Source commit: `5f19a119b53e89e604c34a78ab1c22584f3ff647`
- Command: `python3 scripts/run_bcic2a_degradation.py --config configs/bcic2a_degradation.yaml`
- Config: `configs/bcic2a_degradation.yaml`
- Output directory: `artifacts/bcic2a_degradation/`
- Dataset: MOABB `BNCI2014_001` (BCIC IV 2a), `left_right_hand` paradigm
- Protocol: CSP-LDA and SQI are fit on clean train epochs only; EOG/EMG/drop degradation is injected into test epochs only.
- Completed subjects: 1, 2, 3, 4, 5, 6, 7, 8, 9
- Failures/errors: none recorded by the runner.

## Mid-severity default operating points

Coverage/accuracy columns report default operating-point coverage and accepted accuracy.

| Condition | Forced acc | Mean SQI | Softmax cov/acc | SQI cov/acc | Fusion cov/acc |
| --- | --- | --- | --- | --- | --- |
| clean | 0.855 | 0.379 | 0.889/0.878 | 0.333/0.846 | 0.821/0.889 |
| eog_mid | 0.838 | 0.362 | 0.883/0.868 | 0.274/0.833 | 0.815/0.879 |
| emg_mid | 0.694 | 0.294 | 0.917/0.700 | 0.177/0.872 | 0.758/0.753 |
| drop_mid | 0.532 | 0.365 | 0.904/0.541 | 0.284/0.520 | 0.870/0.544 |

## Artifact inventory

- `RUN_NOTES.md`
- `default_policy_accuracy_by_severity.png`
- `default_policy_coverage_by_severity.png`
- `mid_risk_coverage_curves.png`
- `subject_01.json`
- `subject_01_mid_risk_coverage.png`
- `subject_02.json`
- `subject_02_mid_risk_coverage.png`
- `subject_03.json`
- `subject_03_mid_risk_coverage.png`
- `subject_04.json`
- `subject_04_mid_risk_coverage.png`
- `subject_05.json`
- `subject_05_mid_risk_coverage.png`
- `subject_06.json`
- `subject_06_mid_risk_coverage.png`
- `subject_07.json`
- `subject_07_mid_risk_coverage.png`
- `subject_08.json`
- `subject_08_mid_risk_coverage.png`
- `subject_09.json`
- `subject_09_mid_risk_coverage.png`
- `summary.json`
