from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from rf_drone_detection.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


def main() -> None:
    rng = np.random.default_rng(42)
    signal_dir = PROCESSED_DATA_DIR / "synthetic_smoke"
    signal_dir.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    splits = ["train"] * 24 + ["validation"] * 8 + ["test"] * 8
    labels = ([0, 1] * 20)[: len(splits)]

    for idx, (split, label) in enumerate(zip(splits, labels, strict=True)):
        sample_id = f"synthetic_{idx:03d}"
        recording_id = f"recording_{idx:03d}"
        noise = rng.normal(0, 0.4, size=2048)
        if label:
            t = np.arange(2048)
            carrier = np.sin(2 * np.pi * 0.08 * t)
            envelope = 1.0 + 0.4 * np.sin(2 * np.pi * 0.005 * t)
            values = noise + envelope * carrier
            original_label = "synthetic_drone_associated"
        else:
            values = noise
            original_label = "synthetic_background"

        path = signal_dir / f"{sample_id}.npy"
        np.save(path, values.astype(np.float32))
        rows.append(
            {
                "sample_id": sample_id,
                "source": "synthetic_smoke",
                "recording_id": recording_id,
                "path": str(path),
                "original_label": original_label,
                "binary_label": label,
                "split": split,
                "snr": 10 if label else None,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(INTERIM_DATA_DIR / "synthetic_manifest.csv", index=False)
    print(f"Wrote {len(manifest)} synthetic samples to {signal_dir}")
    print(f"Wrote manifest to {INTERIM_DATA_DIR / 'synthetic_manifest.csv'}")


if __name__ == "__main__":
    main()
