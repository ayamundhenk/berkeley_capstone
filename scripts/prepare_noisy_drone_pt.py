from __future__ import annotations

from pathlib import Path
import argparse
import struct
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from rf_drone_detection.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PipelineConfig
from rf_drone_detection.manifest import create_recording_disjoint_splits


NEGATIVE_LABELS = {"noise"}


def normalize_label(label: object, class_lookup: dict[int, str]) -> str:
    if isinstance(label, str):
        return label
    try:
        return class_lookup[int(label)]
    except Exception:
        return str(label)


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def unpack_dataset(dataset: object) -> tuple[object, object, object | None]:
    """Return samples, labels, and optional SNRs from common torch-saved layouts."""

    if isinstance(dataset, dict):
        sample_key = next(
            (key for key in ["samples", "data", "iq", "IQ", "x", "X", "spectrograms"] if key in dataset),
            None,
        )
        label_key = next((key for key in ["labels", "label", "y", "Y", "class", "class_int"] if key in dataset), None)
        snr_key = next((key for key in ["snr", "SNR", "snrs"] if key in dataset), None)
        if sample_key and label_key:
            return dataset[sample_key], dataset[label_key], dataset.get(snr_key) if snr_key else None

    if isinstance(dataset, (list, tuple)) and dataset and isinstance(dataset[0], dict):
        sample_key = next(
            (key for key in ["sample", "data", "iq", "IQ", "x", "X", "spectrogram"] if key in dataset[0]),
            None,
        )
        label_key = next((key for key in ["label", "y", "class", "class_int"] if key in dataset[0]), None)
        snr_key = next((key for key in ["snr", "SNR"] if key in dataset[0]), None)
        if sample_key and label_key:
            samples = [row[sample_key] for row in dataset]
            labels = [row[label_key] for row in dataset]
            snrs = [row.get(snr_key) for row in dataset] if snr_key else None
            return samples, labels, snrs

    if isinstance(dataset, (list, tuple)) and len(dataset) >= 2:
        samples, labels = dataset[0], dataset[1]
        snrs = dataset[2] if len(dataset) >= 3 else None
        return samples, labels, snrs

    raise ValueError(
        "Could not infer dataset.pt structure. Run this script with --inspect-only and update unpack_dataset()."
    )


def member_data_offset(zip_path: Path, member: str) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member} is compressed; direct memmap extraction requires stored zip members.")
        with zip_path.open("rb") as handle:
            handle.seek(info.header_offset)
            header = handle.read(30)
            fields = struct.unpack("<4s5H3I2H", header)
            filename_length = fields[-2]
            extra_length = fields[-1]
            return info.header_offset + 30 + filename_length + extra_length


def inspect_torch_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            print(f"{info.filename}: size={info.file_size} compress_type={info.compress_type}")
    print(
        "Known Noisy Drone RF layout: x_iq=(98705,2,16384) float32, "
        "x_spec=(98705,2,128,128) float32, y=(98705,) int64, snr=(98705,) int32."
    )


def prepare_known_noisy_drone_layout(args: argparse.Namespace) -> bool:
    with zipfile.ZipFile(args.dataset) as archive:
        names = set(archive.namelist())
    required = {"archive/data/0", "archive/data/2", "archive/data/3"}
    if not required.issubset(names):
        return False

    sample_count = 98705
    x_iq = np.memmap(
        args.dataset,
        dtype=np.float32,
        mode="r",
        offset=member_data_offset(args.dataset, "archive/data/0"),
        shape=(sample_count, 2, 16384),
    )
    labels = np.memmap(
        args.dataset,
        dtype=np.int64,
        mode="r",
        offset=member_data_offset(args.dataset, "archive/data/2"),
        shape=(sample_count,),
    )
    snrs = np.memmap(
        args.dataset,
        dtype=np.int32,
        mode="r",
        offset=member_data_offset(args.dataset, "archive/data/3"),
        shape=(sample_count,),
    )

    class_stats = pd.read_csv(args.class_stats)
    class_lookup = dict(zip(class_stats["class_int"], class_stats["class"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    class_counts: dict[str, int] = {}
    for idx, raw_label in enumerate(labels):
        original_label = normalize_label(raw_label, class_lookup)
        if class_counts.get(original_label, 0) >= args.max_samples_per_class:
            continue

        sample_id = f"noisy_drone_rf_{len(rows):06d}"
        output_path = args.output_dir / f"{sample_id}.npy"
        # Save as (time, I/Q) so the shared feature loader treats it as complex I/Q.
        np.save(output_path, np.asarray(x_iq[idx].T, dtype=np.float32))

        class_counts[original_label] = class_counts.get(original_label, 0) + 1
        rows.append(
            {
                "sample_id": sample_id,
                "source": "noisy_drone_rf",
                "recording_id": sample_id,
                "path": str(output_path),
                "original_label": original_label,
                "binary_label": 0 if original_label.lower() in NEGATIVE_LABELS else 1,
                "split": "",
                "snr": int(snrs[idx]),
            }
        )

        if len(class_counts) >= len(class_lookup) and all(
            class_counts.get(label, 0) >= args.max_samples_per_class for label in class_lookup.values()
        ):
            break

    manifest = create_recording_disjoint_splits(pd.DataFrame(rows), config=PipelineConfig())
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} samples to {args.output_dir}")
    print(f"Wrote split manifest to {args.manifest}")
    print("Class counts:", class_counts)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a bounded manifest-ready subset from Noisy Drone dataset.pt.")
    parser.add_argument("--dataset", type=Path, default=Path("data/external/noisy_drone_rf/dataset.pt"))
    parser.add_argument("--class-stats", type=Path, default=Path("data/external/noisy_drone_rf/class_stats.csv"))
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR / "noisy_drone_rf")
    parser.add_argument("--manifest", type=Path, default=INTERIM_DATA_DIR / "manifest.csv")
    parser.add_argument("--max-samples-per-class", type=int, default=1000)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Missing dataset file: {args.dataset}")

    if args.inspect_only:
        inspect_torch_zip(args.dataset)
        return

    if prepare_known_noisy_drone_layout(args):
        return

    try:
        import torch
    except Exception as exc:
        raise SystemExit("PyTorch is required for unknown dataset.pt layouts.") from exc

    dataset = torch.load(args.dataset, map_location="cpu")

    class_stats = pd.read_csv(args.class_stats)
    class_lookup = dict(zip(class_stats["class_int"], class_stats["class"]))
    samples, labels, snrs = unpack_dataset(dataset)
    labels_array = as_numpy(labels).reshape(-1)
    snr_array = as_numpy(snrs).reshape(-1) if snrs is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    class_counts: dict[str, int] = {}
    for idx, raw_label in enumerate(labels_array):
        original_label = normalize_label(raw_label, class_lookup)
        if class_counts.get(original_label, 0) >= args.max_samples_per_class:
            continue

        sample = samples[idx] if not hasattr(samples, "__getitem__") else samples[idx]
        sample_array = as_numpy(sample).astype(np.float32)
        sample_id = f"noisy_drone_rf_{len(rows):06d}"
        output_path = args.output_dir / f"{sample_id}.npy"
        np.save(output_path, sample_array)

        class_counts[original_label] = class_counts.get(original_label, 0) + 1
        rows.append(
            {
                "sample_id": sample_id,
                "source": "noisy_drone_rf",
                "recording_id": sample_id,
                "path": str(output_path),
                "original_label": original_label,
                "binary_label": 0 if original_label.lower() in NEGATIVE_LABELS else 1,
                "split": "",
                "snr": snr_array[idx] if snr_array is not None else "",
            }
        )

    manifest = create_recording_disjoint_splits(pd.DataFrame(rows), config=PipelineConfig())
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} samples to {args.output_dir}")
    print(f"Wrote split manifest to {args.manifest}")
    print("Class counts:", class_counts)


if __name__ == "__main__":
    main()
