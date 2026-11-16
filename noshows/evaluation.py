"""Measure the model and the real business cost of its decisions."""

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

CALL_COST = 8
MISSED_NO_SHOW_COST = 35
THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 91), 2)


def calculate_cost(actual, calls, call_cost=CALL_COST, miss_cost=MISSED_NO_SHOW_COST):
    """Eight dollars for each call + 35 dollars for each uncalled no-show.

    Hand check: [1,0,1,0] with calls [0,0,1,1] costs 2*8 + 1*35 = 51.
    """
    actual = np.asarray(actual)
    calls = np.asarray(calls)
    if actual.shape != calls.shape:
        raise ValueError("actual and calls must have the same shape")
    if np.any(~np.isin(actual, [0, 1])) or np.any(~np.isin(calls, [0, 1])):
        raise ValueError("actual and calls must only contain 0 and 1")
    if call_cost < 0 or miss_cost < 0:
        raise ValueError("costs cannot be negative")
    return float(
        call_cost * np.count_nonzero(calls)
        + miss_cost * np.sum((actual == 1) & (calls == 0))
    )


def make_calls(probabilities, threshold):
    """At the threshold counts as a call, not as a missed warning."""
    return (np.asarray(probabilities) >= threshold).astype(int)


def choose_threshold(
    actual, probabilities, call_cost=CALL_COST, miss_cost=MISSED_NO_SHOW_COST
):
    """Search 91 thresholds without looking at the final test data.

    When two thresholds cost the same, keep the smallest one (deterministic).
    """
    rows = []
    for threshold in THRESHOLD_GRID:
        decisions = make_calls(probabilities, threshold)
        rows.append(
            {"threshold": float(threshold),
             "training_oof_cost": calculate_cost(
                actual, decisions, call_cost, miss_cost
            ),
             "calls": int(decisions.sum())}
        )
    table = pd.DataFrame(rows)
    best_row = table.loc[table["training_oof_cost"].idxmin()]
    return float(best_row["threshold"]), table


def select_threshold_from_training(pipeline, x_train, y_train, cv):
    """Use predictions for rows each fold's model did NOT train on."""
    probabilities = cross_val_predict(
        pipeline, x_train, y_train, method="predict_proba", cv=cv, n_jobs=1
    )[:, 1]
    return choose_threshold(y_train, probabilities)


def score_predictions(actual, probabilities, threshold):
    """Return five common ML scores and the clinic cost on the SAME rows."""
    decisions = make_calls(probabilities, threshold)
    return {
        "accuracy": float(accuracy_score(actual, decisions)),
        "precision": float(precision_score(actual, decisions, zero_division=0)),
        "recall": float(recall_score(actual, decisions, zero_division=0)),
        "f1": float(f1_score(actual, decisions, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "cost": calculate_cost(actual, decisions),
        "calls": int(decisions.sum()),
    }


def savings_interval(actual, decisions, baseline_calls, seed=11, repeats=2000):
    """Bootstrap a 95% range for savings using the already-finalized test rows.

    We resample appointments, not models. This is descriptive uncertainty, NOT
    proof the result will transfer to new clinics or real patients.
    """
    actual = np.asarray(actual)
    decisions = np.asarray(decisions)
    baseline_calls = np.asarray(baseline_calls)
    rng = np.random.default_rng(seed)
    n = len(actual)
    savings = np.empty(repeats)
    for index in range(repeats):
        sampled = rng.integers(0, n, size=n)
        original = calculate_cost(actual[sampled], baseline_calls[sampled])
        proposed = calculate_cost(actual[sampled], decisions[sampled])
        savings[index] = original - proposed
    return [round(float(x), 2) for x in np.percentile(savings, [2.5, 97.5])]
