# Noisy Drone RF Metadata

Small metadata files were downloaded from Kaggle successfully, but the main `dataset.pt` file is not local yet because the current volume does not have enough free disk space.

Dataset URL: https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification

License reported by Kaggle: CC BY 4.0.

## Available Metadata

Files downloaded:

- `data/external/noisy_drone_rf/SNR_stats.csv`
- `data/external/noisy_drone_rf/class_stats.csv`

Total samples reported by both metadata files: 98,705.

## Class Mapping

| Original Class | Class ID | Count | Binary Label |
| --- | ---: | ---: | ---: |
| DJI | 0 | 2,194 | 1 |
| FutabaT14 | 1 | 6,938 | 1 |
| FutabaT7 | 2 | 3,661 | 1 |
| Graupner | 3 | 6,481 | 1 |
| Noise | 4 | 52,552 | 0 |
| Taranis | 5 | 16,546 | 1 |
| Turnigy | 6 | 10,333 | 1 |

Positive class total: 46,153.

Negative class total: 52,552.

## SNR Coverage

SNR values range from -20 dB to 30 dB in 2 dB steps. Each SNR bin contains approximately 3,794 to 3,800 samples.

## Current Disk Blocker

`dataset.pt` is 25,876,504,771 bytes according to Kaggle. The Kaggle CLI downloads this file as `dataset.pt.zip` first, with a reported transfer size of about 21.9 GiB. The normal download/extract workflow therefore needs enough room for both the zip and extracted `.pt` file, roughly 50 GB of free space.

On the latest attempt, the local volume had about 24 GiB free. The download was intentionally stopped early because it would not have had enough room to extract the dataset.

Once enough disk space is available, download the full file:

```bash
kaggle datasets download sgluege/noisy-drone-rf-signal-classification -f dataset.pt -p data/external/noisy_drone_rf --force
```

Then inspect and prepare it:

```bash
python scripts/prepare_noisy_drone_pt.py --inspect-only
python scripts/prepare_noisy_drone_pt.py --max-samples-per-class 1000
python scripts/run_experiment.py --manifest data/interim/manifest.csv --output-dir reports/artifacts/noisy_drone_rf
```
