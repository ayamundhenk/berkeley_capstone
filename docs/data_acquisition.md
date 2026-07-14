# Data Acquisition Guide

This project is designed around public RF datasets that are downloaded locally but not committed.

## Primary Dataset

Use the Kaggle dataset:

- Noisy Drone RF Signal Classification
- URL: https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification
- Local target: `data/external/noisy_drone_rf/`

Example using the Kaggle CLI:

```bash
pip install kaggle
mkdir -p data/external/noisy_drone_rf
kaggle datasets download -d sgluege/noisy-drone-rf-signal-classification -p data/external/noisy_drone_rf --unzip
```

Kaggle usually requires an API token at `~/.kaggle/kaggle.json`. Do not commit that token.

After download, inspect the extracted directory layout. If it contains one folder per class label, build the manifest with:

```bash
python scripts/build_manifest_from_labeled_dirs.py data/external/noisy_drone_rf --output data/interim/manifest.csv
python scripts/create_recording_disjoint_splits.py data/interim/manifest.csv
```

Review `data/interim/manifest.csv` before training. Confirm that noise/background labels have `binary_label = 0` and all drone-associated RF labels have `binary_label = 1`.

The Kaggle dataset currently includes a large `dataset.pt` file. If using that file, make sure at least 30 GB of disk space is available, then run:

```bash
kaggle datasets download sgluege/noisy-drone-rf-signal-classification -f dataset.pt -p data/external/noisy_drone_rf --force
python scripts/prepare_noisy_drone_pt.py --inspect-only
python scripts/prepare_noisy_drone_pt.py --max-samples-per-class 1000
python scripts/run_experiment.py --manifest data/interim/manifest.csv --output-dir reports/artifacts/noisy_drone_rf
```

## Secondary Dataset

Use DroneRF as a secondary robustness check:

- Paper/dataset description: https://pmc.ncbi.nlm.nih.gov/articles/PMC6727013/
- Local target: `data/external/dronerf/`

DroneRF should be treated as an external-source evaluation because its collection procedure and signal representation differ from the primary dataset. Do not merge it into train/validation/test without documenting the domain shift.

Build an external manifest with `split = external`. If the directory is class-labeled:

```bash
python scripts/build_manifest_from_labeled_dirs.py data/external/dronerf --source dronerf --output data/interim/dronerf_external_manifest.csv
```

Then set the `split` column to `external` if needed:

```bash
python scripts/mark_manifest_external.py data/interim/dronerf_external_manifest.csv
```

If the downloaded DroneRF files are still in `.rar` archives, create a compact real-data subset without unpacking the full dataset:

```bash
python scripts/prepare_dronerf_subset.py
```

This streams a small number of CSV members from the local DroneRF archives, stores compact `.npy` windows in `data/processed/dronerf_subset/`, and writes `data/interim/dronerf_external_manifest.csv`.

For a bounded real-data prototype using DroneRF as the train/validation/test source:

```bash
python scripts/prepare_dronerf_subset.py --split-mode primary --members-per-archive 4 --max-values 8192 --output-dir data/processed/dronerf_primary --manifest data/interim/dronerf_primary_manifest.csv
python scripts/run_experiment.py --manifest data/interim/dronerf_primary_manifest.csv --output-dir reports/artifacts/dronerf_primary
```

Treat these results as DroneRF-prototype findings. They are useful for capstone development, but they do not replace the originally planned Noisy Drone RF primary evaluation unless the project scope is explicitly revised.

## Expected Final Evidence

The final capstone report should not use synthetic smoke-test metrics. It should include:

- Primary held-out test metrics from Noisy Drone RF.
- Validation-selected decision threshold with validation FPR `<= 0.05`.
- Confusion matrix and PR curve for the primary held-out test.
- External DroneRF metrics reported separately as domain-shift evidence.
- Performance slices by SNR, source, or operating mode when metadata is available.
