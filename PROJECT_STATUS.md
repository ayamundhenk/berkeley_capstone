# Project Status

Implemented:

- Reproducible RF-drone detection project scaffold.
- Manifest-first data contract with recording-disjoint split validation.
- Synthetic smoke-data generator for local verification.
- Log-spectrogram preprocessing and engineered spectral features.
- Logistic-regression and random-forest baseline path when `scikit-learn` is installed.
- Dependency-free fallback classifiers for smoke testing when `scikit-learn` is unavailable.
- Optional compact CNN training module when PyTorch is installed.
- External-source evaluation helper for DroneRF-style domain-shift checks.
- DroneRF subset preparation script that streams compact `.npy` windows from local `.rar` archives.
- Experiment runner that writes model comparison, metrics, and predictions into `reports/artifacts/`.
- Noisy Drone RF `dataset.pt` preparation script for bounded extraction after the large Kaggle file is available.
- Validation threshold selection that maximizes recall subject to false-positive rate `<= 0.05`.
- Held-out test evaluation with recall, precision, F1, FPR, PR-AUC, ROC-AUC, confusion counts, and prediction output contract.
- Jupyter notebook skeleton aligned with the capstone report flow.
- Final report outline aligned with the supplied report template.
- Unit tests for leakage detection and FPR-constrained thresholding.
- Compact DroneRF real-data subset at `data/processed/dronerf_subset/`.
- External DroneRF manifest at `data/interim/dronerf_external_manifest.csv`.
- Bounded DroneRF prototype train/validation/test subset at `data/processed/dronerf_primary/`.
- DroneRF prototype experiment artifacts at `reports/artifacts/dronerf_primary/`.
- Noisy Drone RF metadata files at `data/external/noisy_drone_rf/SNR_stats.csv` and `data/external/noisy_drone_rf/class_stats.csv`.
- Noisy Drone RF metadata summary at `reports/noisy_drone_metadata.md`.
- Kaggle credentials were verified and Kaggle dataset access works.
- Noisy Drone RF `dataset.pt` downloaded and extracted at `data/external/noisy_drone_rf/dataset.pt`.
- Balanced Noisy Drone RF primary subset prepared at `data/processed/noisy_drone_rf/`.
- Primary Noisy Drone RF manifest created at `data/interim/manifest.csv`.
- Primary experiment artifacts written to `reports/artifacts/noisy_drone_rf/`.
- Primary results summary written to `reports/noisy_drone_primary_results.md`.
- Report-ready charts written to `reports/artifacts/noisy_drone_rf/charts/`.
- Final report Markdown draft written to `reports/final_report_draft.md`.

Verified locally:

```text
/Users/privateaya_1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall rf_drone_detection scripts tests
/Users/privateaya_1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/create_synthetic_smoke_data.py
/Users/privateaya_1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/run_smoke_pipeline.py
/Users/privateaya_1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests
/Users/privateaya_1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/prepare_dronerf_subset.py --members-per-archive 1 --max-values 8192
/opt/anaconda3/bin/python scripts/prepare_dronerf_subset.py --split-mode primary --members-per-archive 4 --max-values 8192 --output-dir data/processed/dronerf_primary --manifest data/interim/dronerf_primary_manifest.csv
/opt/anaconda3/bin/python scripts/run_experiment.py --manifest data/interim/dronerf_primary_manifest.csv --output-dir reports/artifacts/dronerf_primary
/opt/anaconda3/bin/python scripts/prepare_noisy_drone_pt.py --max-samples-per-class 1000
/opt/anaconda3/bin/python scripts/run_experiment.py --manifest data/interim/manifest.csv --external-manifest data/interim/dronerf_external_manifest.csv --output-dir reports/artifacts/noisy_drone_rf
/opt/anaconda3/bin/python scripts/summarize_results.py --artifact-dir reports/artifacts/noisy_drone_rf
/opt/anaconda3/bin/python scripts/create_report_charts.py --artifact-dir reports/artifacts/noisy_drone_rf
```

Additional verification:

- External evaluation plumbing was run on the compact DroneRF subset using the synthetic smoke-trained model. This verifies the code path, but those metrics are not valid capstone findings.
- The DroneRF prototype experiment ran with the intended `scikit-learn` logistic regression/random forest baselines using local Anaconda Python. The selected model was logistic regression. On the tiny four-sample held-out test set, it produced FPR `0.0`, recall `0.0`, PR-AUC `1.0`, and ROC-AUC `1.0` at threshold `0.9836878532870398`. These are prototype/plumbing results only because the subset is intentionally small.
- The primary Noisy Drone RF experiment selected logistic regression. Held-out test metrics: recall `0.645`, precision `0.9910371318822023`, F1 `0.7814235234729934`, FPR `0.035`, PR-AUC `0.9843906885542705`, ROC-AUC `0.9097291666666667`, threshold `0.5297634093511037`.

Current blocker for final capstone results:

- No implementation blocker remains for the planned reproducible notebook/report workflow.
- Remaining optional polish: transfer `reports/final_report_draft.md` into the provided Word template if a `.docx` submission is required, and decide whether to expand the DroneRF external subset beyond the tiny plumbing sample.

Next required steps:

1. Run or export the notebook with `manifest_path = INTERIM_DATA_DIR / "manifest.csv"`.
2. Optionally transfer `reports/final_report_draft.md` into the provided Word template.
3. Optionally expand the compact DroneRF external subset for a stronger domain-shift evaluation.
