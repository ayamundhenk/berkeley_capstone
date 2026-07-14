from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from rf_drone_detection.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from rf_drone_detection.config import PipelineConfig
from rf_drone_detection.manifest import create_recording_disjoint_splits


DEFAULT_ARCHIVES = [
    (
        "background",
        0,
        Path("/Users/privateaya_1/Downloads/DroneRF/Background RF activites/RF Data_00000_L1.rar"),
    ),
    (
        "background",
        0,
        Path("/Users/privateaya_1/Downloads/DroneRF/Background RF activites/RF Data_00000_L2.rar"),
    ),
    (
        "background",
        0,
        Path("/Users/privateaya_1/Downloads/DroneRF/Background RF activites/RF Data_00000_H1.rar"),
    ),
    (
        "bepop_drone",
        1,
        Path("/Users/privateaya_1/Downloads/DroneRF/Bepop drone/RF Data_10000_L.rar"),
    ),
    (
        "ar_drone",
        1,
        Path("/Users/privateaya_1/Downloads/DroneRF/AR drone/RF Data_10100_L.rar"),
    ),
    (
        "phantom_drone",
        1,
        Path("/Users/privateaya_1/Downloads/DroneRF/Phantom drone/RF Data_11000_L1.rar"),
    ),
]


def list_csv_members(archive: Path) -> list[str]:
    output = subprocess.check_output(["bsdtar", "-tf", str(archive)], text=True, stderr=subprocess.DEVNULL)
    return [line for line in output.splitlines() if line.endswith(".csv")]


def read_member_values(archive: Path, member: str, max_values: int) -> np.ndarray:
    process = subprocess.Popen(
        ["bsdtar", "-xOf", str(archive), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdout is not None
    values: list[float] = []

    for line in process.stdout:
        for token in line.strip().split(","):
            if not token:
                continue
            values.append(float(token))
            if len(values) >= max_values:
                process.kill()
                return np.asarray(values, dtype=np.float32)

    process.wait()
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact DroneRF external-evaluation subset.")
    parser.add_argument("--members-per-archive", type=int, default=2)
    parser.add_argument("--max-values", type=int, default=16384)
    parser.add_argument("--split-mode", choices=["external", "primary"], default="external")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR / "dronerf_subset")
    parser.add_argument("--manifest", type=Path, default=INTERIM_DATA_DIR / "dronerf_external_manifest.csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_name, binary_label, archive in DEFAULT_ARCHIVES:
        if not archive.exists():
            print(f"Skipping missing archive: {archive}")
            continue

        members = list_csv_members(archive)[: args.members_per_archive]
        for member in members:
            sample_id = f"dronerf_{len(rows):04d}"
            values = read_member_values(archive, member, args.max_values)
            output_path = args.output_dir / f"{sample_id}.npy"
            np.save(output_path, values)
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": "dronerf",
                    "recording_id": Path(member).stem,
                    "path": str(output_path),
                    "original_label": label_name,
                    "binary_label": binary_label,
                    "split": "external" if args.split_mode == "external" else "",
                    "notes": f"streamed_first_{len(values)}_values_from_{archive.name}:{member}",
                }
            )

    if not rows:
        raise SystemExit("No DroneRF archive members were prepared.")

    manifest = pd.DataFrame(rows)
    if args.split_mode == "primary":
        manifest = create_recording_disjoint_splits(
            manifest,
            config=PipelineConfig(random_state=42, test_size=0.2, validation_size=0.2),
        )
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} DroneRF subset samples to {args.output_dir}")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
