from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PipelineConfig


REQUIRED_COLUMNS = {
    "sample_id",
    "source",
    "recording_id",
    "path",
    "original_label",
    "binary_label",
    "split",
}


@dataclass(frozen=True)
class SplitSummary:
    split: str
    rows: int
    positives: int
    negatives: int
    recordings: int


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if manifest["sample_id"].duplicated().any():
        duplicates = manifest.loc[manifest["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values found: {duplicates[:5]}")

    invalid_labels = set(manifest["binary_label"].dropna().unique()).difference({0, 1})
    if invalid_labels:
        raise ValueError(f"binary_label must contain only 0/1 values; found {sorted(invalid_labels)}")

    invalid_splits = set(manifest["split"].dropna().unique()).difference({"train", "validation", "test", "external"})
    if invalid_splits:
        raise ValueError(f"split must be train, validation, test, or external; found {sorted(invalid_splits)}")

    split_counts = manifest.groupby("recording_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        examples = leaking.index.astype(str).tolist()[:5]
        raise ValueError(f"Recording IDs appear in multiple splits: {examples}")


def summarize_splits(manifest: pd.DataFrame) -> list[SplitSummary]:
    summaries: list[SplitSummary] = []
    for split, group in manifest.groupby("split", sort=True):
        summaries.append(
            SplitSummary(
                split=split,
                rows=len(group),
                positives=int(group["binary_label"].sum()),
                negatives=int((group["binary_label"] == 0).sum()),
                recordings=group["recording_id"].nunique(),
            )
        )
    return summaries


def create_recording_disjoint_splits(
    manifest: pd.DataFrame,
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Assign train/validation/test splits while keeping recording IDs disjoint."""

    config = config or PipelineConfig()
    working = manifest.copy()
    if "split" not in working.columns:
        working["split"] = ""

    recordings = (
        working.groupby("recording_id", as_index=False)
        .agg(binary_label=("binary_label", "max"), source=("source", "first"))
        .sort_values("recording_id")
    )

    train_val_ids, test_ids = _train_test_split(
        recordings,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    train_val = recordings[recordings["recording_id"].isin(train_val_ids)]
    validation_fraction = config.validation_size / (1 - config.test_size)
    train_ids, validation_ids = _train_test_split(
        train_val,
        test_size=validation_fraction,
        random_state=config.random_state,
    )

    split_map = {recording_id: "train" for recording_id in train_ids}
    split_map.update({recording_id: "validation" for recording_id in validation_ids})
    split_map.update({recording_id: "test" for recording_id in test_ids})
    working["split"] = working["recording_id"].map(split_map)
    validate_manifest(working)
    return working


def _train_test_split(recordings: pd.DataFrame, test_size: float, random_state: int) -> tuple[list[str], list[str]]:
    rng = pd.Series(recordings["recording_id"].unique()).sample(frac=1.0, random_state=random_state).tolist()
    labels = recordings.set_index("recording_id")["binary_label"].to_dict()
    train_ids: list[str] = []
    test_ids: list[str] = []

    for label in sorted(set(labels.values())):
        label_ids = [recording_id for recording_id in rng if labels[recording_id] == label]
        n_test = max(1, round(len(label_ids) * test_size)) if len(label_ids) > 1 else 0
        test_ids.extend(label_ids[:n_test])
        train_ids.extend(label_ids[n_test:])

    return train_ids, test_ids
