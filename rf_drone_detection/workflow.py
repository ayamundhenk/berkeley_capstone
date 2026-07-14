from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .evaluation import average_precision, choose_threshold_for_recall_at_fpr, evaluate_scores
from .features import build_feature_matrix
from .manifest import load_manifest, validate_manifest
from .models import train_logistic_regression, train_random_forest


def run_baseline_workflow(manifest_path: str | Path, config: PipelineConfig | None = None) -> dict[str, object]:
    config = config or PipelineConfig()
    manifest = load_manifest(manifest_path)
    manifest = manifest[manifest["split"].isin(["train", "validation", "test"])].copy()
    validate_manifest(manifest)

    x, _ = build_feature_matrix(manifest["path"].tolist(), config=config)
    y = manifest["binary_label"].to_numpy()

    train_mask = manifest["split"].eq("train").to_numpy()
    validation_mask = manifest["split"].eq("validation").to_numpy()
    test_mask = manifest["split"].eq("test").to_numpy()

    candidates = [
        train_logistic_regression(x[train_mask], y[train_mask], random_state=config.random_state),
        train_random_forest(x[train_mask], y[train_mask], random_state=config.random_state),
    ]

    model_rows = []
    selected = None
    selected_threshold = None
    selected_key = None

    for fitted in candidates:
        validation_scores = fitted.predict_proba(x[validation_mask])
        threshold = choose_threshold_for_recall_at_fpr(
            y[validation_mask],
            validation_scores,
            max_fpr=config.max_validation_fpr,
        )
        pr_auc = average_precision(y[validation_mask], validation_scores)
        key = (threshold.recall, pr_auc, threshold.f1)
        model_rows.append(
            {
                "model": fitted.name,
                "validation_threshold": threshold.threshold,
                "validation_recall": threshold.recall,
                "validation_precision": threshold.precision,
                "validation_fpr": threshold.fpr,
                "validation_f1": threshold.f1,
                "validation_pr_auc": pr_auc,
            }
        )
        if selected is None or key > selected_key:
            selected = fitted
            selected_threshold = threshold
            selected_key = key

    assert selected is not None
    assert selected_threshold is not None

    test_scores = selected.predict_proba(x[test_mask])
    test_metrics = evaluate_scores(y[test_mask], test_scores, selected_threshold.threshold)

    prediction_columns = [
        column
        for column in ["sample_id", "binary_label", "source", "recording_id", "original_label", "snr", "operating_mode"]
        if column in manifest.columns
    ]
    predictions = manifest.loc[test_mask, prediction_columns].copy()
    predictions["p_drone_associated"] = test_scores
    predictions["predicted_label"] = (test_scores >= selected_threshold.threshold).astype(int)
    predictions["decision_threshold"] = selected_threshold.threshold

    return {
        "model_comparison": pd.DataFrame(model_rows).sort_values(
            ["validation_recall", "validation_pr_auc", "validation_f1"],
            ascending=False,
        ),
        "selected_model": selected.name,
        "selected_estimator": selected,
        "selected_threshold": selected_threshold,
        "test_metrics": test_metrics,
        "test_predictions": predictions,
    }


def evaluate_external_manifest(
    fitted_model: object,
    threshold: float,
    manifest_path: str | Path,
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    """Evaluate a trained model on an external manifest without retraining."""

    config = config or PipelineConfig()
    manifest = load_manifest(manifest_path)
    external = manifest[manifest["split"].eq("external")].copy()
    if external.empty:
        raise ValueError("External manifest must contain at least one row with split='external'.")

    x_external, _ = build_feature_matrix(external["path"].tolist(), config=config)
    y_external = external["binary_label"].to_numpy()
    scores = fitted_model.predict_proba(x_external)
    metrics = evaluate_scores(y_external, scores, threshold)

    prediction_columns = [
        column
        for column in ["sample_id", "binary_label", "source", "recording_id", "original_label", "snr", "operating_mode"]
        if column in external.columns
    ]
    predictions = external[prediction_columns].copy()
    predictions["p_drone_associated"] = scores
    predictions["predicted_label"] = (scores >= threshold).astype(int)
    predictions["decision_threshold"] = threshold

    return {
        "external_metrics": metrics,
        "external_predictions": predictions,
    }
