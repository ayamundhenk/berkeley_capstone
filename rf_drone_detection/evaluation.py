from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    recall: float
    precision: float
    fpr: float
    f1: float


def choose_threshold_for_recall_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.05,
) -> ThresholdResult:
    thresholds = np.unique(np.r_[0.0, y_score, 1.0])
    best: ThresholdResult | None = None

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if fpr > max_fpr:
            continue

        result = ThresholdResult(
            threshold=float(threshold),
            recall=float(recall_from_counts(tp, fn)),
            precision=float(precision_from_counts(tp, fp)),
            fpr=float(fpr),
            f1=float(f1_from_counts(tp, fp, fn)),
        )
        if best is None or (result.recall, result.f1, result.precision) > (best.recall, best.f1, best.precision):
            best = result

    if best is None:
        return ThresholdResult(threshold=1.0, recall=0.0, precision=0.0, fpr=0.0, f1=0.0)
    return best


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    metrics = {
        "threshold": float(threshold),
        "recall": float(recall_from_counts(tp, fn)),
        "precision": float(precision_from_counts(tp, fp)),
        "f1": float(f1_from_counts(tp, fp, fn)),
        "fpr": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "pr_auc": float(average_precision(y_true, y_score)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def grouped_metrics(frame: pd.DataFrame, group_column: str, threshold: float) -> pd.DataFrame:
    rows = []
    for group_value, group in frame.groupby(group_column):
        metrics = evaluate_scores(
            group["binary_label"].to_numpy(),
            group["p_drone_associated"].to_numpy(),
            threshold,
        )
        metrics[group_column] = group_value
        metrics["rows"] = len(group)
        rows.append(metrics)
    return pd.DataFrame(rows)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def precision_from_counts(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall_from_counts(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = precision_from_counts(tp, fp)
    recall = recall_from_counts(tp, fn)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    positives = y_sorted.sum()
    if positives == 0:
        return 0.0
    cumulative_tp = np.cumsum(y_sorted)
    precision = cumulative_tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / positives)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    wins = 0.0
    for positive in positives:
        wins += float((positive > negatives).sum())
        wins += 0.5 * float((positive == negatives).sum())
    return wins / (len(positives) * len(negatives))
