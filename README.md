# RF-Based Drone Signal Detection

## Project overview

Drones can be difficult to identify visually when they are far away, obstructed, or operating in poor lighting. This project examines whether patterns in radio and radar signals can help distinguish drone-associated activity from background or non-UAV activity.

The primary research question is:

> Can a machine-learning model identify drone-associated RF signals while keeping false alarms low?

The project began with CardRF as the proposed data source. Because that dataset was not reliably accessible, the primary analysis changed to the public [Noisy Drone RF Signal Classification dataset](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification).

## Technical notebooks

1. [Primary RF analysis and EDA](notebooks/rf_drone_detection_capstone.executed.ipynb)
   Data cleaning, exploratory analysis, feature engineering, logistic regression baseline, error analysis, and performance by signal-to-noise ratio.

2. [Final model comparison](notebooks/final_model_comparison.executed.ipynb)
   Logistic regression and random forest comparison, five-fold cross-validation, grid-search tuning, and held-out evaluation on a separate synthetic micro-Doppler benchmark.

Source versions without stored output are also available in the [notebooks directory](notebooks/).

## Data

### Primary evidence: Noisy Drone RF

The primary public dataset contains 98,705 examples from six consumer drone/controller signal classes plus noise. A manageable 7,000-example subset was used:

- 1,000 noise examples
- 6,000 drone-associated examples
- Signal-to-noise ratio (SNR) from -20 dB to 30 dB

The target is binary:

- 0: background/noise
- 1: drone-associated RF activity

Raw signal files are too large for this repository. Compact predictions, metrics, and charts are included under `reports/artifacts/noisy_drone_rf/`.

### Secondary methodological benchmark

The final model-selection notebook uses the public [Micro-Doppler Aerial Classification Dataset](https://www.kaggle.com/datasets/mithula05/micro-doppler-aerial-classification-dataset). It contains 2,800 synthetic radar time series representing birds, drones, aircraft, and stealth UAVs.

This source is used only to demonstrate multiple-model comparison, cross-validation, and grid search. It is synthetic radar data—not real RF communication data—so its scores are reported separately and are not treated as proof of real-world RF performance.

## Data preparation

The primary workflow:

1. Checks required fields, missing values, duplicate sample IDs, and valid probability ranges.
2. Converts raw I/Q signal values into normalized log-spectrograms.
3. Extracts ten numerical spectral summaries for the RF baseline.
4. Uses separate training, validation, and test sets.
5. Selects the prediction threshold with validation data.
6. Evaluates the selected baseline on held-out test data.

The secondary benchmark:

1. Checks 2,800 rows for missing values, duplicates, and invalid labels.
2. Reshapes each row into 100 time steps for amplitude, velocity, and energy.
3. Engineers 24 summary features using means, standard deviations, ranges, quartiles, and root mean square values.
4. Creates an 80/20 stratified train/test split.
5. Uses five-fold stratified cross-validation and grid search on the training portion.

## Models and evaluation

### Primary RF baseline

Logistic regression is used as the initial RF classifier. Recall is the main metric because it measures how many actual drone-associated signals are detected. False-positive rate is also important because an awareness system that repeatedly flags background noise would not be useful.

The validation threshold was selected to maximize recall while keeping the validation false-positive rate at or below 5%.

### Final comparison

The secondary benchmark compares:

- Logistic regression
- Random forest

Grid search tests regularization and class weighting for logistic regression, and tree count, depth, leaf size, and class weighting for random forest. Parameter choices are evaluated with five-fold cross-validation using recall.

### Model considered but not selected

YOLO was considered as a possible advanced model, but it was not selected. YOLO is designed primarily to locate objects within images using bounding boxes. This project classifies complete RF and micro-Doppler signals and does not contain bounding-box annotations. A spectrogram classification CNN would be a more appropriate future deep-learning option, but adding one was outside the scope needed to answer the current research question.

## Findings

### Primary real RF result

The logistic regression baseline achieved:

| Metric | Result |
| --- | ---: |
| Recall | 64.5% |
| Precision | 99.1% |
| F1 score | 78.1% |
| False-positive rate | 3.5% |
| PR-AUC | 98.4% |

The model produced 774 true positives, 193 true negatives, 7 false positives, and 426 false negatives.

The most important finding is that signal quality strongly affects detection. Recall was poor for weak signals at negative SNR values and improved sharply as SNR increased. From approximately 6 dB upward, recall was generally strong in the held-out sample.

### Secondary synthetic benchmark

Both tuned models achieved 100% five-fold cross-validation recall and 100% held-out recall. Logistic regression produced one false positive and no missed drone/UAV examples.

These near-perfect results should be interpreted cautiously. The benchmark is synthetic, balanced, and generated with strongly separated class patterns. The scores demonstrate the modeling workflow more than they demonstrate real-world readiness. Logistic regression is preferred because it matches random forest performance with a simpler explanation.

## Nontechnical conclusion

The primary RF model is good at avoiding false alarms: when it reports drone-associated activity, it is usually correct. Its weakness is missed detections, especially when the signal is weak. That tradeoff makes the model a useful academic baseline but not a system ready for operational use.

The synthetic benchmark shows that common classifiers can separate clearly defined drone-like signal patterns, but it does not remove the need for testing with real RF recordings.

## Recommendations and next steps

1. Reacquire or regenerate the primary RF feature matrix and repeat the same five-fold grid-search comparison on real RF data.
2. Focus improvement efforts on low-SNR examples, where most missed detections occur.
3. Collect more real recordings across devices, locations, interference conditions, and frequency bands.
4. Keep recording sessions separated across training and test sets to reduce leakage.
5. Calibrate the decision threshold for the cost of missed detections versus false alarms.
6. Validate the selected model on a larger external dataset before considering deployment.

## Limitations

- The primary subset is balanced by original label and does not reflect real-world prevalence.
- Public laboratory data may not represent unseen hardware or environments.
- RF-silent or autonomous drones may not produce detectable communication signals.
- The secondary benchmark is synthetic radar data and cannot validate the real RF detector.
- This project is an academic prototype for defensive awareness, not a safety or enforcement system.

## Repository organization

```text
.
├── notebooks/
│   ├── rf_drone_detection_capstone.ipynb
│   ├── rf_drone_detection_capstone.executed.ipynb
│   ├── final_model_comparison.ipynb
│   └── final_model_comparison.executed.ipynb
├── reports/
│   └── artifacts/
│       ├── noisy_drone_rf/
│       └── micro_doppler/
├── rf_drone_detection/
├── scripts/
├── tests/
├── README.md
└── requirements.txt
```

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project tests:

```bash
python -m unittest discover -s tests
```

Regenerate the secondary model artifacts after placing `astra_dataset.csv` under `data/external/micro_doppler/`:

```bash
python scripts/run_final_model_comparison.py
```
