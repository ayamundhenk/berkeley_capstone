from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from rf_drone_detection.config import INTERIM_DATA_DIR, PROJECT_ROOT, PipelineConfig
from rf_drone_detection.workflow import evaluate_external_manifest, run_baseline_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RF drone detection experiment and write report artifacts.")
    parser.add_argument("--manifest", type=Path, default=INTERIM_DATA_DIR / "manifest.csv")
    parser.add_argument("--external-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "artifacts")
    args = parser.parse_args()

    config = PipelineConfig(sample_rate_hz=1.0, spectrogram_nperseg=128, spectrogram_noverlap=64)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_baseline_workflow(args.manifest, config=config)
    result["model_comparison"].to_csv(args.output_dir / "model_comparison.csv", index=False)
    result["test_predictions"].to_csv(args.output_dir / "test_predictions.csv", index=False)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(result["test_metrics"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Selected model:", result["selected_model"])
    print("Test metrics:")
    print(json.dumps(result["test_metrics"], indent=2, sort_keys=True))

    if args.external_manifest:
        external = evaluate_external_manifest(
            result["selected_estimator"],
            result["selected_threshold"].threshold,
            args.external_manifest,
            config=config,
        )
        external["external_predictions"].to_csv(args.output_dir / "external_predictions.csv", index=False)
        (args.output_dir / "external_metrics.json").write_text(
            json.dumps(external["external_metrics"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("External metrics:")
        print(json.dumps(external["external_metrics"], indent=2, sort_keys=True))

    artifacts = [
        {
            "artifact": "model_comparison",
            "path": str(args.output_dir / "model_comparison.csv"),
        },
        {
            "artifact": "test_metrics",
            "path": str(args.output_dir / "test_metrics.json"),
        },
        {
            "artifact": "test_predictions",
            "path": str(args.output_dir / "test_predictions.csv"),
        },
    ]
    if args.external_manifest:
        artifacts.extend(
            [
                {
                    "artifact": "external_metrics",
                    "path": str(args.output_dir / "external_metrics.json"),
                },
                {
                    "artifact": "external_predictions",
                    "path": str(args.output_dir / "external_predictions.csv"),
                },
            ]
        )
    summary = pd.DataFrame(artifacts)
    summary.to_csv(args.output_dir / "artifact_manifest.csv", index=False)


if __name__ == "__main__":
    main()
