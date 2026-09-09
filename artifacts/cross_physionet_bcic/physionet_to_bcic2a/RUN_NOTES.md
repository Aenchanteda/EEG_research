# Cross-dataset PhysioNet/BCIC RUN_NOTES

Config: `configs/cross_physionet_to_bcic2a.yaml`
Direction: PhysionetMI -> BNCI2014_001

## Protocol

- Binary CSP-LDA only; no EEGNet, FBCNet, or domain-adversarial model is used.
- Source subjects are pooled into one labeled training set.
- Target labels are used only for evaluation, never for SQI fitting, threshold fitting, or model training.
- SQI is fit on the source training epochs and transformed on target test epochs.
- Locked policy IDs: forced, softmax, margin, sqi, combined_and, fusion.

## Harmonization

- Labels: left_hand vs right_hand only.
- PhysioNet: MOABB PhysionetMI with imagined=True and executed=False, selecting imagery left/right fist runs 4, 8, and 12 through the dataset wrapper.
- Channels: deterministic intersection in BCIC IV 2a 22-channel order.
- Selected channels (22): Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz.
- Epoch window: 0.5 to 3.0 s.
- Common sampling frequency: 160.0 Hz.

## Subject counts

- Source configured subjects: 108
- Source run limit: none
- Source loaded subjects: 108 ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
- Source configured exclusions: [{'subject': 88, 'reason': 'MOABB documents subject 88 at 128 Hz; full multi-subject loads use 160 Hz subjects only.'}]
- Target configured subjects: 9
- Target run limit: none
- Target loaded subjects: 9 ([1, 2, 3, 4, 5, 6, 7, 8, 9])
- Target configured exclusions: []
- PhysioNet subject 88 is excluded by config for full-dataset loads because MOABB documents it at 128 Hz rather than 160 Hz.

## Pooled target metrics

- Accuracy: 0.674
- ECE: 0.037
- Brier: 0.418

## Default policy table

| Policy | Coverage | Accepted accuracy | Risk |
| --- | ---: | ---: | ---: |
| forced | 1.000 | 0.674 | 0.326 |
| softmax | 0.577 | 0.750 | 0.250 |
| margin | 0.577 | 0.750 | 0.250 |
| sqi | 0.490 | 0.696 | 0.304 |
| combined_and | 0.288 | 0.784 | 0.216 |
| fusion | 0.656 | 0.718 | 0.282 |

## BCIC IV 2a within-subject reference

- Source: PR #2 BCIC IV 2a within-subject CSP-LDA artifacts
- Accuracy: 0.855
- ECE: 0.049
- Brier: 0.207
- Cross-dataset accuracy is expected to drop under montage, subject-population, and collection-protocol shift.

## Previous partial-run comparison

| Run | Source subjects | Target subjects | Accuracy | ECE | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous partial | 12 | 9 | 0.622 | 0.120 | 0.487 |
| Expanded current | 108 | 9 | 0.674 | 0.037 | 0.418 |

## Skips

- source: none
- target: none
