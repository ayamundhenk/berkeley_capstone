from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from rf_drone_detection.config import PipelineConfig
from rf_drone_detection.manifest import create_recording_disjoint_splits, summarize_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign recording-disjoint train/validation/test splits.")
    parser.add_argument("manifest", type=Path, help="Input manifest CSV.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV. Defaults to overwriting input.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    split_manifest = create_recording_disjoint_splits(
        manifest,
        config=PipelineConfig(
            random_state=args.seed,
            test_size=args.test_size,
            validation_size=args.validation_size,
        ),
    )

    output = args.output or args.manifest
    split_manifest.to_csv(output, index=False)
    print(f"Wrote split manifest to {output}")
    for summary in summarize_splits(split_manifest):
        print(summary)


if __name__ == "__main__":
    main()
