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
- Source run limit: 12
- Source loaded subjects: 12 ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
- Target configured subjects: 9
- Target run limit: none
- Target loaded subjects: 9 ([1, 2, 3, 4, 5, 6, 7, 8, 9])
- PhysioNet subject 88 is excluded by config for full-dataset loads because MOABB documents it at 128 Hz rather than 160 Hz.

## Pooled target metrics

- Accuracy: 0.622
- ECE: 0.120
- Brier: 0.487

## Default policy table

| Policy | Coverage | Accepted accuracy | Risk |
| --- | ---: | ---: | ---: |
| forced | 1.000 | 0.622 | 0.378 |
| softmax | 0.794 | 0.647 | 0.353 |
| margin | 0.794 | 0.647 | 0.353 |
| sqi | 0.501 | 0.633 | 0.367 |
| combined_and | 0.390 | 0.662 | 0.338 |
| fusion | 0.833 | 0.645 | 0.355 |

## BCIC IV 2a within-subject reference

- Source: PR #2 BCIC IV 2a within-subject CSP-LDA artifacts
- Accuracy: 0.855
- ECE: 0.049
- Brier: 0.207
- Cross-dataset accuracy is expected to drop under montage, subject-population, and collection-protocol shift.

## Skips

- source: none
- target: none
