"""Model evaluation and appointment outreach cost calculations."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict


def calculate_cost(y_true, y_pred) -> float:
    """Calculate outreach and missed no-show costs."""

    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    false_negatives = np.sum((actual == 1) & (predicted == 0))
    flagged = np.sum(predicted == 1)

    return float((flagged * 8) + (false_negatives * 35))


def evaluate_predictions(y_true, y_pred, y_probability) -> dict:
    """Calculate classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "cost": calculate_cost(y_true, y_pred),
    }


def get_threshold_predictions(probabilities, threshold: float) -> np.ndarray:
    """Convert predicted probabilities into binary decisions."""

    return (np.asarray(probabilities) >= threshold).astype(int)


def find_best_threshold(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> tuple[float, pd.DataFrame]:
    """Select a threshold using out-of-fold training probabilities."""

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    results = []

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = get_threshold_predictions(
            probabilities,
            threshold,
        )

        results.append(
            {
                "threshold": round(float(threshold), 2),
                "cost": calculate_cost(y_train, predictions),
            }
        )

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["cost"].idxmin()]

    return float(best_row["threshold"]), results_df
