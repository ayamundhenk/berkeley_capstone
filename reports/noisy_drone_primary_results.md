# Noisy Drone RF Primary Results

This report summarizes the primary capstone experiment using the Kaggle Noisy Drone RF Signal Classification dataset.

## Data

- Dataset: Noisy Drone RF Signal Classification.
- Local source file: `data/external/noisy_drone_rf/dataset.pt`.
- Prepared subset: `data/processed/noisy_drone_rf/`.
- Manifest: `data/interim/manifest.csv`.
- Samples used: 7,000, balanced at 1,000 per original class.
- Binary mapping: `Noise = 0`; DJI, FutabaT14, FutabaT7, Graupner, Taranis, and Turnigy = `1`.
- SNR range: -20 dB to 30 dB.

## Split Summary

| Split | Background | Drone-Associated | Total |
| --- | ---: | ---: | ---: |
| Train | 600 | 3,600 | 4,200 |
| Validation | 200 | 1,200 | 1,400 |
| Test | 200 | 1,200 | 1,400 |

## Model Selection

The validation objective was maximum recall subject to false-positive rate `<= 0.05`, with PR-AUC as the tie-breaker.

| Model | Validation Threshold | Validation Recall | Validation Precision | Validation FPR | Validation F1 | Validation PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.5297634093511037 | 0.6733333333333333 | 0.9877750611246944 | 0.05 | 0.800792864222002 | 0.9805050388441733 |
| Random forest | 0.9517403730612136 | 0.6658333333333334 | 0.9876390605686032 | 0.05 | 0.7954206072672971 | 0.9854064347067697 |

Selected model: logistic regression.

## Held-Out Test Results

| Metric | Value |
| --- | ---: |
| Threshold | 0.5297634093511037 |
| Recall | 0.645 |
| Precision | 0.9910371318822023 |
| F1 | 0.7814235234729934 |
| False-positive rate | 0.035 |
| PR-AUC | 0.9843906885542705 |
| ROC-AUC | 0.9097291666666667 |
| True negatives | 193 |
| False positives | 7 |
| False negatives | 426 |
| True positives | 774 |

## Performance Pattern

The selected threshold is conservative. It keeps false positives below the 5% cap on both validation and held-out test, but this trades away recall at low SNR.

Recall by SNR is weakest from -20 dB through about -6 dB, improves sharply around -4 dB to 4 dB, and is near-perfect from roughly 6 dB upward. This is a report-worthy limitation: the detector is reliable in cleaner signal conditions but misses many low-SNR drone-associated samples.

Recall by original class ranged from about 0.617 to 0.700 across drone-associated classes in the held-out test. Noise false-positive rate was 0.035.

## External DroneRF Check

A tiny DroneRF external subset was evaluated as a domain-shift plumbing check. At the Noisy Drone RF validation-selected threshold, external recall was 0.0 with FPR 0.0. Because the subset has only four samples and comes from a different data-generation process, this should be reported as evidence of domain shift and workflow readiness, not as a definitive external benchmark.

## Artifacts

Primary artifacts are in `reports/artifacts/noisy_drone_rf/`:

- `model_comparison.csv`
- `test_metrics.json`
- `test_predictions.csv`
- `metrics_by_original_label.csv`
- `metrics_by_snr.csv`
- `external_metrics.json`
- `external_predictions.csv`
- `charts/model_comparison.png`
- `charts/confusion_matrix.png`
- `charts/metrics_by_snr.png`
- `charts/recall_by_original_label.png`
