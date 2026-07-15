# RF-Based Drone Signal Detection Capstone

## Project summary

This capstone asks whether a machine-learning model can tell the difference between drone-associated radio-frequency signals and background noise. The main submission notebook is written as a step-by-step analysis and uses two standard classification models: logistic regression and random forest.

The project originally proposed CardRF, but that dataset was not reliably accessible. The primary source was changed to the public Noisy Drone RF Signal Classification dataset while keeping the same research question.

Start here:

- `notebooks/rf_drone_detection_capstone.executed.ipynb`: completed Module 20 notebook with outputs
- `reports/final_report_draft.md`: draft content for the final non-technical report

This project implements a reproducible binary classifier for RF-based drone detection. The target is:

- `1`: drone-associated RF activity
- `0`: RF background/noise

The implementation follows the capstone plan: public RF datasets, a manifest-driven workflow, log-spectrogram features, classical baselines, a compact CNN path, validation-threshold selection, and final reporting focused on recall with a maximum validation false-positive rate of 5%.

## Repository layout

```text
.
├── data/
│   ├── external/        # raw downloaded datasets, not committed
│   ├── interim/         # manifests and derived metadata
│   └── processed/       # model-ready arrays, not committed
├── notebooks/
│   └── rf_drone_detection_capstone.ipynb
├── docs/
│   └── data_acquisition.md
├── reports/
│   └── final_report_outline.md
├── rf_drone_detection/
│   ├── config.py
│   ├── evaluation.py
│   ├── features.py
│   ├── manifest.py
│   ├── models.py
│   └── workflow.py
├── scripts/
│   ├── build_manifest_from_labeled_dirs.py
│   ├── create_report_charts.py
│   ├── create_recording_disjoint_splits.py
│   ├── create_synthetic_smoke_data.py
│   ├── mark_manifest_external.py
│   ├── prepare_noisy_drone_pt.py
│   ├── prepare_dronerf_subset.py
│   ├── run_experiment.py
│   └── run_smoke_pipeline.py
└── tests/
    └── test_workflow.py
```

## Data sources

Primary dataset:

- [Noisy Drone RF Signal Classification](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification), used for training and primary evaluation. Map all drone/control-signal classes to `drone_associated=1` and the noise/background class to `0`.

Secondary dataset:

- [DroneRF](https://pmc.ncbi.nlm.nih.gov/articles/PMC6727013/), used for data-composition analysis and external robustness checks. Because its collection setup differs from the primary data, report it as a domain-shift evaluation rather than mixing it blindly with the primary dataset.

Place raw data under `data/external/`. The code expects a manifest with the columns described below.

See [data_acquisition.md](/Users/privateaya_1/Documents/berkeley_capstone/docs/data_acquisition.md) for dataset download and manifest setup details.

For class-labeled folders, start with:

```bash
python scripts/build_manifest_from_labeled_dirs.py data/external/noisy_drone_rf --output data/interim/manifest.csv
python scripts/create_recording_disjoint_splits.py data/interim/manifest.csv
```

For the Kaggle `dataset.pt` layout, after downloading the full file:

```bash
python scripts/prepare_noisy_drone_pt.py --inspect-only
python scripts/prepare_noisy_drone_pt.py --max-samples-per-class 1000
python scripts/run_experiment.py --manifest data/interim/manifest.csv --output-dir reports/artifacts/noisy_drone_rf
python scripts/summarize_results.py --artifact-dir reports/artifacts/noisy_drone_rf
python scripts/create_report_charts.py --artifact-dir reports/artifacts/noisy_drone_rf
```

## Manifest contract

The project is manifest-first. Each row should represent one model input sample:

```text
sample_id, source, recording_id, path, original_label, binary_label, split
```

Optional columns such as `snr`, `operating_mode`, `device`, and `notes` are preserved and used for grouped reporting when present.

Splits must be recording-disjoint: all windows or augmentations from the same `recording_id` stay in exactly one split.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/create_synthetic_smoke_data.py
python scripts/run_smoke_pipeline.py
python -m unittest discover -s tests
```

The synthetic data is only a smoke-test fixture. It proves the workflow runs before the public RF datasets are downloaded.

For a bounded local DroneRF prototype:

```bash
python scripts/prepare_dronerf_subset.py --split-mode primary --members-per-archive 4 --max-values 8192 --output-dir data/processed/dronerf_primary --manifest data/interim/dronerf_primary_manifest.csv
python scripts/run_experiment.py --manifest data/interim/dronerf_primary_manifest.csv --output-dir reports/artifacts/dronerf_primary
```

## Modeling approach

The baseline path extracts spectral summary features and trains:

- logistic regression
- random forest

The CNN path trains a compact image classifier on normalized log-spectrograms through `rf_drone_detection.cnn.train_compact_cnn`. If PyTorch is not installed, the project still runs the classical baselines and reports the CNN dependency gap honestly.

The selected decision threshold is chosen on validation data to maximize recall subject to false-positive rate `<= 0.05`. PR-AUC is used as the tie-breaker when models satisfy the FPR constraint.

## Ethical use

This is a research prototype for defensive RF awareness and academic study. It should not be represented as a deployed security system, a real-time enforcement tool, or a reliable detector outside the measured RF conditions. Results should explicitly state limitations around hardware, environment, RF silence, encryption, geography, and dataset shift.
