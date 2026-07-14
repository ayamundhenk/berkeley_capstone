# RF-Based Drone Detection and Classification Final Report Draft

## 1. Define Problem Statement

The goal of this project is to detect drone-associated radio-frequency activity from RF signal samples. This is useful because drones may be difficult to identify visually in cluttered, low-light, or long-distance environments, while their controllers and communication links can leave measurable RF patterns.

The project focuses on binary detection:

- `1`: drone-associated RF signal
- `0`: RF background/noise

The main technical challenge is separating drone-associated signals from noise across different signal-to-noise ratios. The model must also avoid excessive false positives, because a detector that constantly alerts on background RF activity would be difficult to trust.

## 2. Model Outcomes or Predictions

This is a supervised binary classification problem. Each sample receives:

- `p_drone_associated`: estimated probability that the sample contains drone-associated RF activity
- `predicted_label`: thresholded binary prediction
- `decision_threshold`: selected validation threshold

The threshold was selected on validation data to maximize drone recall while keeping validation false-positive rate at or below 5%.

## 3. Data Acquisition

The primary dataset is the Kaggle Noisy Drone RF Signal Classification dataset. The downloaded metadata reports 98,705 total examples across seven original classes:

- DJI
- FutabaT14
- FutabaT7
- Graupner
- Noise
- Taranis
- Turnigy

`Noise` was mapped to the negative class. The six remaining classes were mapped to the positive drone-associated class.

The full `dataset.pt` file was downloaded and a balanced subset was prepared for this capstone run:

- 1,000 samples per original class
- 7,000 total samples
- SNR range from -20 dB to 30 dB
- Recording-disjoint train, validation, and test splits

DroneRF was also used as a small external-domain check. Because DroneRF is collected and represented differently, it was not mixed into the primary train/test split.

## 4. Data Preprocessing

The PyTorch archive was inspected and read directly as a stored zip archive. The primary input used `x_iq`, which has shape `(98705, 2, 16384)` in the full dataset. Each selected sample was saved as a `.npy` I/Q array.

Signals were converted into normalized log-spectrograms. Classical baseline models used compact spectral summary features extracted from those spectrograms.

The final split sizes were:

| Split | Background | Drone-Associated | Total |
| --- | ---: | ---: | ---: |
| Train | 600 | 3,600 | 4,200 |
| Validation | 200 | 1,200 | 1,400 |
| Test | 200 | 1,200 | 1,400 |

## 5. Modeling

Two baseline models were trained:

| Model | Validation Threshold | Validation Recall | Validation Precision | Validation FPR | Validation F1 | Validation PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.5297634093511037 | 0.6733333333333333 | 0.9877750611246944 | 0.05 | 0.800792864222002 | 0.9805050388441733 |
| Random forest | 0.9517403730612136 | 0.6658333333333334 | 0.9876390605686032 | 0.05 | 0.7954206072672971 | 0.9854064347067697 |

Logistic regression was selected because it achieved the highest validation recall while satisfying the 5% false-positive-rate cap.

## 6. Model Evaluation

Held-out test results for the selected logistic regression model:

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

The model met the false-positive requirement on the held-out test set, with FPR of 3.5%. The tradeoff is recall: the model missed 426 of 1,200 drone-associated test samples.

The strongest performance pattern was by SNR. Recall was very low at the lowest SNR values and improved sharply as SNR increased. From roughly 6 dB upward, recall was near-perfect. This indicates that the model is much more reliable when RF signal quality is moderate or strong.

The DroneRF external check produced recall of 0.0 on the tiny external subset at the Noisy Drone RF threshold. This should be interpreted as a domain-shift warning, not a definitive external benchmark.

## Limitations

This is a research prototype, not a deployed drone-detection system. It should not be assumed to detect:

- RF-silent drones
- drone systems outside the represented classes
- unsupported frequency bands
- environments unlike the public datasets
- low-SNR conditions where the current model has poor recall

The external DroneRF analysis should be expanded before making strong claims about generalization.

## Reproducibility

Key artifacts:

- `data/interim/manifest.csv`
- `reports/artifacts/noisy_drone_rf/model_comparison.csv`
- `reports/artifacts/noisy_drone_rf/test_metrics.json`
- `reports/artifacts/noisy_drone_rf/test_predictions.csv`
- `reports/artifacts/noisy_drone_rf/metrics_by_snr.csv`
- `reports/artifacts/noisy_drone_rf/metrics_by_original_label.csv`
- `reports/artifacts/noisy_drone_rf/charts/`

Main commands:

```bash
python scripts/prepare_noisy_drone_pt.py --max-samples-per-class 1000
python scripts/run_experiment.py --manifest data/interim/manifest.csv --external-manifest data/interim/dronerf_external_manifest.csv --output-dir reports/artifacts/noisy_drone_rf
python scripts/summarize_results.py --artifact-dir reports/artifacts/noisy_drone_rf
python scripts/create_report_charts.py --artifact-dir reports/artifacts/noisy_drone_rf
```
