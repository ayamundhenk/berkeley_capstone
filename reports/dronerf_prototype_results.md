# DroneRF Prototype Results

This report captures a bounded real-data prototype run using local DroneRF `.rar` archives. It is useful as implementation evidence, but it should not be presented as the final capstone result because the original plan still calls for Noisy Drone RF Signal Classification as the primary dataset.

## Data

- Source: local DroneRF archives from `/Users/privateaya_1/Downloads/DroneRF`.
- Prepared manifest: `data/interim/dronerf_primary_manifest.csv`.
- Prepared signal windows: `data/processed/dronerf_primary/`.
- Samples: 24 compact `.npy` windows, streamed from archive CSV members.
- Labels: background/noise = `0`, drone-associated RF = `1`.
- Splits: recording-disjoint train/validation/test.

## Model Selection

Artifacts are in `reports/artifacts/dronerf_primary/`.

Validation comparison:

| Model | Validation Threshold | Validation Recall | Validation Precision | Validation FPR | Validation F1 | Validation PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.9836878532870398 | 0.5 | 1.0 | 0.0 | 0.6666666666666666 | 0.8333333333333333 |
| random_forest | 0.98 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5833333333333333 |

Selected model: logistic regression.

## Held-Out Test Metrics

The selected validation threshold was `0.9836878532870398`, chosen to preserve validation FPR `<= 0.05`.

| Metric | Value |
| --- | ---: |
| Recall | 0.0 |
| Precision | 0.0 |
| F1 | 0.0 |
| FPR | 0.0 |
| PR-AUC | 1.0 |
| ROC-AUC | 1.0 |
| True Negative | 2 |
| False Positive | 0 |
| False Negative | 2 |
| True Positive | 0 |

## Interpretation

This run proves the real-data preparation, manifest validation, feature extraction, model training, threshold selection, and artifact-writing workflow. The threshold is very conservative because the validation set is tiny and the project priority is reliable drone recall subject to a strict false-positive cap.

The result should be treated as a prototype. The next scientifically meaningful step is to run the same workflow on the planned Noisy Drone RF primary dataset and then use DroneRF as the external/domain-shift check.
