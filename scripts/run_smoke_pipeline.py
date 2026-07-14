from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rf_drone_detection.config import INTERIM_DATA_DIR, PipelineConfig
from rf_drone_detection.workflow import run_baseline_workflow


def main() -> None:
    result = run_baseline_workflow(
        INTERIM_DATA_DIR / "synthetic_manifest.csv",
        config=PipelineConfig(sample_rate_hz=1.0, spectrogram_nperseg=128, spectrogram_noverlap=64),
    )
    print("Model comparison")
    print(result["model_comparison"].to_string(index=False))
    print("\nSelected model:", result["selected_model"])
    print("Test metrics")
    for key, value in result["test_metrics"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
