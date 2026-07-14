# RF-Based Drone Signal Detection Final Report Outline

## 1. Define Problem Statement

Detect whether an RF sample contains drone-associated activity or background/noise. The core challenge is identifying useful signal patterns across SNR variation while avoiding excessive false alarms.

## 2. Model Outcomes or Predictions

This is a supervised binary classification task. The model outputs `p_drone_associated` and a thresholded label. The threshold is selected on validation data to prioritize recall while keeping false-positive rate at or below 5%.

## 3. Data Acquisition

Primary data comes from Noisy Drone RF Signal Classification. Secondary analysis uses DroneRF as an external robustness check. The report should include dataset provenance, licensing, class mapping, source differences, class balance, SNR distribution, and representative spectrograms.

## 4. Data Preprocessing

Raw I/Q or signal vectors are converted to normalized log-spectrograms. Splits are recording-disjoint so related windows or augmentations do not leak across train, validation, and test sets.

## 5. Modeling

Train logistic regression and random forest baselines on engineered spectral features. Train a compact CNN on spectrograms when the environment supports PyTorch. Compare all models on validation recall, FPR, PR-AUC, and F1.

## 6. Model Evaluation

Evaluate the selected model once on held-out test data. Report recall, precision, F1, PR-AUC, ROC-AUC, FPR, confusion matrix, threshold behavior, and performance by SNR and source.

## Limitations

The model should be framed as a dataset-bound research prototype. It cannot detect RF-silent drones, unsupported frequencies, unsupported hardware, or conditions outside the represented public datasets.
