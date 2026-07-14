from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rf_drone_detection.config import PipelineConfig
from rf_drone_detection.evaluation import choose_threshold_for_recall_at_fpr
from rf_drone_detection.manifest import validate_manifest
from rf_drone_detection.workflow import evaluate_external_manifest, run_baseline_workflow


class WorkflowTests(unittest.TestCase):
    def test_manifest_rejects_recording_leakage(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "source": "unit",
                    "recording_id": "r1",
                    "path": "a.npy",
                    "original_label": "noise",
                    "binary_label": 0,
                    "split": "train",
                },
                {
                    "sample_id": "b",
                    "source": "unit",
                    "recording_id": "r1",
                    "path": "b.npy",
                    "original_label": "drone",
                    "binary_label": 1,
                    "split": "test",
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_manifest(manifest)

    def test_threshold_respects_false_positive_rate_constraint(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.01, 0.02, 0.04, 0.9, 0.35, 0.6, 0.8, 0.95])

        result = choose_threshold_for_recall_at_fpr(y_true, y_score, max_fpr=0.05)

        self.assertLessEqual(result.fpr, 0.05)
        self.assertGreater(result.threshold, 0.9)

    def test_external_manifest_evaluates_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = []
            splits = ["train"] * 12 + ["validation"] * 4 + ["test"] * 4 + ["external"] * 4
            labels = ([0, 1] * 12)[: len(splits)]

            for idx, (split, label) in enumerate(zip(splits, labels, strict=True)):
                values = np.zeros(512, dtype=np.float32)
                if label:
                    values += np.sin(2 * np.pi * 0.08 * np.arange(512)).astype(np.float32)
                path = root / f"sample_{idx:03d}.npy"
                np.save(path, values)
                rows.append(
                    {
                        "sample_id": f"sample_{idx:03d}",
                        "source": "unit",
                        "recording_id": f"recording_{idx:03d}",
                        "path": str(path),
                        "original_label": "drone" if label else "background",
                        "binary_label": label,
                        "split": split,
                    }
                )

            manifest = pd.DataFrame(rows)
            primary_manifest = root / "primary.csv"
            external_manifest = root / "external.csv"
            manifest[manifest["split"].ne("external")].to_csv(primary_manifest, index=False)
            manifest[manifest["split"].eq("external")].to_csv(external_manifest, index=False)

            result = run_baseline_workflow(
                primary_manifest,
                config=PipelineConfig(spectrogram_nperseg=64, spectrogram_noverlap=32),
            )
            external = evaluate_external_manifest(
                result["selected_estimator"],
                result["selected_threshold"].threshold,
                external_manifest,
                config=PipelineConfig(spectrogram_nperseg=64, spectrogram_noverlap=32),
            )

            self.assertIn("external_metrics", external)
            self.assertIn("external_predictions", external)
            self.assertEqual(len(external["external_predictions"]), 4)


if __name__ == "__main__":
    unittest.main()
