from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from rf_drone_detection.config import INTERIM_DATA_DIR


BACKGROUND_LABELS = {"noise", "background", "rf_background", "no_drone", "negative"}


def infer_binary_label(label: str) -> int:
    normalized = label.lower().replace(" ", "_").replace("-", "_")
    return 0 if normalized in BACKGROUND_LABELS else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest from class-labeled signal directories.")
    parser.add_argument("data_root", type=Path, help="Directory containing one subdirectory per original label.")
    parser.add_argument("--source", default="noisy_drone_rf", help="Dataset/source name for the manifest.")
    parser.add_argument("--output", type=Path, default=INTERIM_DATA_DIR / "manifest.csv")
    args = parser.parse_args()

    rows = []
    for label_dir in sorted(path for path in args.data_root.iterdir() if path.is_dir()):
        original_label = label_dir.name
        binary_label = infer_binary_label(original_label)
        for path in sorted(label_dir.rglob("*")):
            if path.suffix.lower() not in {".npy", ".csv", ".txt"}:
                continue
            sample_id = f"{args.source}_{len(rows):06d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": args.source,
                    "recording_id": path.stem,
                    "path": str(path),
                    "original_label": original_label,
                    "binary_label": binary_label,
                    "split": "",
                }
            )

    if not rows:
        raise SystemExit(f"No .npy, .csv, or .txt signal files found under {args.data_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} manifest rows to {args.output}")
    print("Review labels before creating final train/validation/test splits.")


if __name__ == "__main__":
    main()
