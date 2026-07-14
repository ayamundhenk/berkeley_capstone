from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from rf_drone_detection.evaluation import evaluate_scores


def write_grouped_metrics(predictions: pd.DataFrame, group_column: str, output: Path) -> None:
    rows = []
    for group_value, group in predictions.groupby(group_column, dropna=False):
        threshold = float(group["decision_threshold"].iloc[0])
        metrics = evaluate_scores(
            group["binary_label"].to_numpy(),
            group["p_drone_associated"].to_numpy(),
            threshold,
        )
        metrics[group_column] = group_value
        metrics["rows"] = len(group)
        rows.append(metrics)
    pd.DataFrame(rows).sort_values(group_column).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready summary tables from experiment artifacts.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions_path = args.artifact_dir / "test_predictions.csv"
    metrics_path = args.artifact_dir / "test_metrics.json"
    predictions = pd.read_csv(predictions_path)

    if "original_label" in predictions.columns:
        write_grouped_metrics(predictions, "original_label", args.artifact_dir / "metrics_by_original_label.csv")
    if "snr" in predictions.columns:
        write_grouped_metrics(predictions, "snr", args.artifact_dir / "metrics_by_snr.csv")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in metrics.items()
        ]
    )
    summary.to_csv(args.artifact_dir / "test_metrics_summary.csv", index=False)
    print(f"Wrote summary tables to {args.artifact_dir}")


if __name__ == "__main__":
    main()
