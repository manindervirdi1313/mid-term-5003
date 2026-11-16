"""Run with ``python -m noshows.train`` to rebuild ALL model files.

We compare on TRAINING folds, pick the cutoff from TRAINING folds, and only
then unlock the held-out test rows once. Imagine a final exam: don't peek!
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from noshows.data import make_appointments
from noshows.evaluation import (
    calculate_cost,
    make_calls,
    savings_interval,
    score_predictions,
    select_threshold_from_training,
)
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    SEED,
    build_pipeline,
    candidate_models,
    split_appointments,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SCORING = {
    "accuracy": "accuracy",
    # The Dummy model never predicts a miss, so define precision as zero.
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def save_json(path, content):
    """Save plain numbers and text so classmates can read the results."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    table = make_appointments(seed=SEED)
    x_train, x_test, y_train, y_test = split_appointments(table)
    print(f"Appointments: {len(table)} | train: {len(x_train)} | test: {len(x_test)}")
    print("The 1,000 test rows are locked until the final evaluation.\n")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Comparing several models on exactly the same five TRAINING folds is fair.
    comparisons = []
    for name, classifier in candidate_models().items():
        scores = cross_validate(
            build_pipeline(classifier), x_train, y_train, cv=cv, scoring=SCORING,
            n_jobs=1,
        )
        item = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        comparisons.append(item)
        print(f"Compared {name}: AUC {item['roc_auc_mean']:.3f}")
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(REPORTS / "cv_results.csv", index=False)

    # Choose logistic regression for interpretability: its AUC is comparable
    # with the strongest candidate (difference no larger than 1 fold std).
    logistic = comparison.loc[comparison["model"] == "Logistic regression"].iloc[0]
    winner = comparison.loc[comparison["roc_auc_mean"].idxmax()]
    within_spread = (
        winner["roc_auc_mean"] - logistic["roc_auc_mean"]
        <= winner["roc_auc_std"]
    )
    selection_reason = (
        "Logistic regression is easy to explain and its ROC-AUC is within one "
        "five-fold standard deviation of the top candidate."
        if within_spread
        else "Logistic regression was selected for interpretability, despite a "
        "larger-than-one-fold-SD ROC-AUC gap; this trade-off is disclosed."
    )

    # There are 3 * 2 = 6 parameter combinations. Still NO test rows.
    search = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=2000, random_state=SEED)),
        param_grid={
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        cv=cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_
    tuned_gain = float(search.best_score_ - logistic["roc_auc_mean"])
    spread = float(logistic["roc_auc_std"])
    tuning_conclusion = (
        "The tuning gain is larger than the untuned model's fold-to-fold spread."
        if tuned_gain > spread
        else "Tuning did not beat the untuned model by more than its "
        "fold-to-fold spread."
    )
    print(f"\nBest parameters: {search.best_params_}")
    print(
        f"Tuned ROC-AUC: {search.best_score_:.4f}; "
        f"untuned: {logistic['roc_auc_mean']:.4f}"
    )

    # Every training row gets a prediction from a fold that did not see it.
    threshold, cutoff_table = select_threshold_from_training(
        model, x_train, y_train, cv
    )
    cutoff_table.to_csv(REPORTS / "threshold_search.csv", index=False)
    print(f"Out-of-fold, cost-selected threshold: {threshold:.2f}")

    # Deliberately BAD experiment: the fee is recorded AFTER the visit. This
    # demonstrates leakage on training folds only; we never ship this model.
    leaked_training = x_train.assign(
        fee_charged=table.loc[x_train.index, "fee_charged"]
    )
    leaky_auc = cross_validate(
        build_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED),
            numeric_features=[*NUMERIC_FEATURES, "fee_charged"],
        ),
        leaked_training, y_train, cv=cv, scoring="roc_auc", n_jobs=1,
    )["test_score"]
    print(f"Honest ROC-AUC: {logistic['roc_auc_mean']:.4f}")
    print(f"LEAKING fee_charged ROC-AUC: {np.mean(leaky_auc):.4f} (NOT shipped)")

    # THE FIRST AND ONLY TEST-SET EVALUATION starts here. No retuning after it.
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_calls = make_calls(test_probabilities, threshold)
    metrics = score_predictions(y_test, test_probabilities, threshold)
    nobody = np.zeros(len(y_test), dtype=int)
    everybody = np.ones(len(y_test), dtype=int)
    costs = {
        "call_nobody": calculate_cost(y_test, nobody),
        "call_everybody": calculate_cost(y_test, everybody),
        "our_model": metrics["cost"],
    }
    baseline = min(costs["call_nobody"], costs["call_everybody"])
    costs["savings_vs_cheaper_baseline"] = baseline - costs["our_model"]
    interval = savings_interval(y_test, test_calls, nobody, seed=SEED)
    interval_everybody = savings_interval(y_test, test_calls, everybody, seed=SEED)
    print(f"\nTest metrics: {metrics}")
    print(f"Costs for 1,000 appointments: {costs}")
    print(f"95% bootstrap savings range vs calling nobody: {interval}")
    print(f"95% bootstrap savings range vs calling everybody: {interval_everybody}")

    # Save the whole pipeline, not just the classifier, so app inputs get the
    # EXACT SAME cleaning as training inputs.
    joblib.dump(model, MODELS / "no_show.joblib")
    info = {
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "model_choice": "Logistic regression (GridSearchCV)",
        "selection_reason": selection_reason,
        "grid_scoring": "roc_auc",
        "grid_combinations": 6,
        "best_settings": search.best_params_,
        "tuned_cv_roc_auc": float(search.best_score_),
        "untuned_cv_roc_auc": float(logistic["roc_auc_mean"]),
        "untuned_cv_roc_auc_std": spread,
        "tuning_gain": tuned_gain,
        "tuning_conclusion": tuning_conclusion,
        "honest_roc_auc_mean": float(logistic["roc_auc_mean"]),
        "leaky_roc_auc_mean": float(np.mean(leaky_auc)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_no_show_count": int(y_test.sum()),
        "test_metrics": metrics,
        "test_costs": costs,
        "bootstrap_95pct_savings_vs_nobody": interval,
        "bootstrap_95pct_savings_vs_everybody": interval_everybody,
        "beats_both_baselines": metrics["cost"] < baseline,
    }
    save_json(MODELS / "model_info.json", info)
    save_json(REPORTS / "metrics.json", {
        "test_metrics": metrics,
        "cost_comparison": costs,
        "bootstrap_95pct_savings_vs_nobody": interval,
        "bootstrap_95pct_savings_vs_everybody": interval_everybody,
        "leakage_experiment": {
            "honest_cv_auc": float(logistic["roc_auc_mean"]),
            "after_visit_fee_cv_auc": float(np.mean(leaky_auc)),
        },
        "tuning_conclusion": tuning_conclusion,
    })
    print("\nSaved model, metadata, comparison, threshold search, and test report.")


if __name__ == "__main__":
    main()
