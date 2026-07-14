from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:
    RandomForestClassifier = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None


@dataclass
class FittedModel:
    name: str
    estimator: object
    validation_pr_auc: float | None = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(x)[:, 1]


def train_logistic_regression(x_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> FittedModel:
    if LogisticRegression is None:
        estimator = NearestCentroidProbability(scale=True)
        estimator.fit(x_train, y_train)
        return FittedModel(name="logistic_regression_fallback", estimator=estimator)

    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    estimator.fit(x_train, y_train)
    return FittedModel(name="logistic_regression", estimator=estimator)


def train_random_forest(x_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> FittedModel:
    if RandomForestClassifier is None:
        estimator = NearestCentroidProbability(scale=False)
        estimator.fit(x_train, y_train)
        return FittedModel(name="random_forest_fallback", estimator=estimator)

    estimator = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    estimator.fit(x_train, y_train)
    return FittedModel(name="random_forest", estimator=estimator)


def pytorch_available() -> bool:
    from .cnn import torch_available

    return torch_available()


class NearestCentroidProbability:
    """Small dependency-free probability model for smoke tests."""

    def __init__(self, scale: bool = True) -> None:
        self.scale = scale

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroidProbability":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y).astype(int)
        self.mean_ = x.mean(axis=0) if self.scale else np.zeros(x.shape[1])
        self.std_ = x.std(axis=0) + 1e-8 if self.scale else np.ones(x.shape[1])
        x_scaled = (x - self.mean_) / self.std_
        self.negative_centroid_ = x_scaled[y == 0].mean(axis=0)
        self.positive_centroid_ = x_scaled[y == 1].mean(axis=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_scaled = (x - self.mean_) / self.std_
        distance_positive = np.linalg.norm(x_scaled - self.positive_centroid_, axis=1)
        distance_negative = np.linalg.norm(x_scaled - self.negative_centroid_, axis=1)
        score = distance_negative - distance_positive
        probability_positive = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - probability_positive, probability_positive])
