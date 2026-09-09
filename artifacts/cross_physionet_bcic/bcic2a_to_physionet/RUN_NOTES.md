# Cross-dataset PhysioNet/BCIC RUN_NOTES

Config: `configs/cross_bcic2a_to_physionet.yaml`
Direction: BNCI2014_001 -> PhysionetMI

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

- Source configured subjects: 9
- Source run limit: none
- Source loaded subjects: 9 ([1, 2, 3, 4, 5, 6, 7, 8, 9])
- Target configured subjects: 108
- Target run limit: 6
- Target loaded subjects: 6 ([1, 2, 3, 4, 5, 6])
- PhysioNet subject 88 is excluded by config for full-dataset loads because MOABB documents it at 128 Hz rather than 160 Hz.

## Pooled target metrics

- Accuracy: 0.530
- ECE: 0.137
- Brier: 0.535

## Default policy table

| Policy | Coverage | Accepted accuracy | Risk |
| --- | ---: | ---: | ---: |
| forced | 1.000 | 0.530 | 0.470 |
| softmax | 0.678 | 0.552 | 0.448 |
| margin | 0.678 | 0.552 | 0.448 |
| sqi | 0.115 | 0.581 | 0.419 |
| combined_and | 0.081 | 0.636 | 0.364 |
| fusion | 0.344 | 0.548 | 0.452 |

## BCIC IV 2a within-subject reference

- Source: PR #2 BCIC IV 2a within-subject CSP-LDA artifacts
- Accuracy: 0.855
- ECE: 0.049
- Brier: 0.207
- Cross-dataset accuracy is expected to drop under montage, subject-population, and collection-protocol shift.

## Skips

- source: none
- target: none
