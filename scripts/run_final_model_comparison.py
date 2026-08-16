from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "external" / "micro_doppler" / "astra_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "artifacts" / "micro_doppler"
RANDOM_STATE = 42


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize 100 time steps for amplitude, velocity, and energy."""

    signal_columns = [str(i) for i in range(300)]
    values = raw[signal_columns].to_numpy(dtype=float).reshape(-1, 100, 3)
    channels = ["amplitude", "velocity", "energy"]
    engineered: dict[str, np.ndarray] = {}

    for index, channel in enumerate(channels):
        channel_values = values[:, :, index]
        engineered[f"{channel}_mean"] = channel_values.mean(axis=1)
        engineered[f"{channel}_std"] = channel_values.std(axis=1)
        engineered[f"{channel}_min"] = channel_values.min(axis=1)
        engineered[f"{channel}_max"] = channel_values.max(axis=1)
        engineered[f"{channel}_median"] = np.median(channel_values, axis=1)
        engineered[f"{channel}_q25"] = np.quantile(channel_values, 0.25, axis=1)
        engineered[f"{channel}_q75"] = np.quantile(channel_values, 0.75, axis=1)
        engineered[f"{channel}_rms"] = np.sqrt(np.mean(channel_values**2, axis=1))

    features = pd.DataFrame(engineered)
    features["original_label"] = raw["label"].astype(int)
    # 1=Drone and 3=Stealth UAV; 0=Bird and 2=Aircraft.
    features["uav_associated"] = raw["label"].isin([1, 3]).astype(int)
    return features


def model_searches() -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return {
        "logistic_regression": GridSearchCV(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
            scoring="recall",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        ),
        "random_forest": GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [150, 300],
                "max_depth": [None, 10],
                "min_samples_leaf": [1, 2],
                "class_weight": [None, "balanced"],
            },
            scoring="recall",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        ),
    }


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"Missing source dataset: {RAW_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    cleaned = raw.drop_duplicates().dropna().reset_index(drop=True)
    features = engineer_features(cleaned)
    features.to_csv(OUTPUT_DIR / "astra_summary_features.csv", index=False)

    x = features.drop(columns=["original_label", "uav_associated"])
    y = features["uav_associated"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    comparison_rows = []
    prediction_frames = []
    best_params: dict[str, object] = {}
    fitted_searches = model_searches()

    for model_name, search in fitted_searches.items():
        search.fit(x_train, y_train)
        probabilities = search.best_estimator_.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        comparison_rows.append(
            {
                "model": model_name,
                "best_cv_recall": search.best_score_,
                "test_recall": recall_score(y_test, predictions),
                "test_precision": precision_score(y_test, predictions),
                "test_f1": f1_score(y_test, predictions),
                "test_pr_auc": average_precision_score(y_test, probabilities),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        best_params[model_name] = search.best_params_
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "actual": y_test.to_numpy(),
                    "predicted": predictions,
                    "probability_uav": probabilities,
                }
            )
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["best_cv_recall", "test_pr_auc"], ascending=False
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    (OUTPUT_DIR / "best_parameters.json").write_text(json.dumps(best_params, indent=2) + "\n")

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_data = comparison.melt(
        id_vars="model",
        value_vars=["best_cv_recall", "test_recall"],
        var_name="evaluation",
        value_name="recall",
    )
    sns.barplot(data=plot_data, x="model", y="recall", hue="evaluation", ax=ax)
    ax.set(
        title="Cross-validation and held-out recall by model",
        xlabel="Model",
        ylabel="Recall",
        ylim=(0, 1.05),
    )
    ax.legend(title="Evaluation", labels=["5-fold CV", "Held-out test"])
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    selected_name = comparison.iloc[0]["model"]
    selected_predictions = predictions[predictions["model"].eq(selected_name)]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        selected_predictions["actual"],
        selected_predictions["predicted"],
        display_labels=["Bird / aircraft", "Drone / UAV"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Held-out confusion matrix: {selected_name.replace('_', ' ').title()}")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(comparison.to_string(index=False))
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    main()
