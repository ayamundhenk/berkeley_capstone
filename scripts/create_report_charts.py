from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd


def save_model_comparison(artifact_dir: Path, chart_dir: Path) -> None:
    data = pd.read_csv(artifact_dir / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data["model"], data["validation_recall"], label="Recall")
    ax.scatter(data["model"], data["validation_fpr"], color="tab:red", label="FPR", zorder=3)
    ax.axhline(0.05, color="tab:red", linestyle="--", linewidth=1, label="FPR cap")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation metric")
    ax.set_title("Validation model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def save_confusion_matrix(artifact_dir: Path, chart_dir: Path) -> None:
    metrics = json.loads((artifact_dir / "test_metrics.json").read_text(encoding="utf-8"))
    matrix = pd.DataFrame(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        index=["Actual background", "Actual drone"],
        columns=["Predicted background", "Predicted drone"],
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix.iloc[row, col]), ha="center", va="center", color="black")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title("Held-out test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(chart_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_recall_by_snr(artifact_dir: Path, chart_dir: Path) -> None:
    data = pd.read_csv(artifact_dir / "metrics_by_snr.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data["snr"], data["recall"], marker="o", label="Recall")
    ax.plot(data["snr"], data["fpr"], marker="o", label="FPR")
    ax.axhline(0.05, color="tab:red", linestyle="--", linewidth=1, label="FPR cap")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Metric")
    ax.set_title("Held-out performance by SNR")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(chart_dir / "metrics_by_snr.png", dpi=160)
    plt.close(fig)


def save_recall_by_label(artifact_dir: Path, chart_dir: Path) -> None:
    data = pd.read_csv(artifact_dir / "metrics_by_original_label.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(data["original_label"], data["recall"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Original class")
    ax.set_ylabel("Recall")
    ax.set_title("Held-out recall by original class")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(chart_dir / "recall_by_original_label.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready PNG charts from experiment artifacts.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--chart-dir", type=Path, default=None)
    args = parser.parse_args()

    chart_dir = args.chart_dir or args.artifact_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    save_model_comparison(args.artifact_dir, chart_dir)
    save_confusion_matrix(args.artifact_dir, chart_dir)
    save_recall_by_snr(args.artifact_dir, chart_dir)
    save_recall_by_label(args.artifact_dir, chart_dir)
    print(f"Wrote charts to {chart_dir}")


if __name__ == "__main__":
    main()
