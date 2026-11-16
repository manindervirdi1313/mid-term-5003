"""Train, compare, tune, and evaluate no-show prediction models."""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from noshows.data import make_appointments
from noshows.evaluation import (
    calculate_cost,
    evaluate_predictions,
    find_best_threshold,
    get_threshold_predictions,
)
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    SEED,
    TARGET,
    build_model_pipeline,
    get_candidate_models,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    """Run the complete model training and evaluation workflow."""

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Generate the dataset and select booking-time features.
    data = make_appointments()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET]

    # 2. Create a stratified 75/25 train-test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    print("Dataset:", data.shape)
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Training no-show rate:", round(y_train.mean(), 4))
    print()

    # 3. Set up five-fold stratified cross-validation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    # 4. Compare a dummy baseline and four candidate models.
    candidate_models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        **get_candidate_models(),
    }

    comparison_rows = []

    for model_name, classifier in candidate_models.items():
        print(f"Cross-validating: {model_name}")

        pipeline = build_model_pipeline(classifier)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )

        row = {"model": model_name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())

        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nCross-validation results:")
    print(comparison.round(4).to_string(index=False))

    # 5. Tune logistic regression using training data only.
    tuning_pipeline = build_model_pipeline(
        LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        )
    )

    parameter_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=tuning_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )

    print("\nRunning hyperparameter tuning...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("Best parameters:", search.best_params_)
    print("Best cross-validation ROC-AUC:", round(search.best_score_, 4))

    # 6. Select the decision threshold using training out-of-fold predictions.
    best_threshold, threshold_results = find_best_threshold(
        best_model,
        X_train,
        y_train,
        cv,
    )

    threshold_results.to_csv(
        REPORTS_DIR / "threshold_search.csv",
        index=False,
    )

    print("\nSelected threshold:", best_threshold)

    # 7. Evaluate the final model on the held-out test set.
    # This is the first use of X_test and y_test.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]

    test_predictions = get_threshold_predictions(
        test_probabilities,
        best_threshold,
    )

    model_metrics = evaluate_predictions(
        y_test,
        test_predictions,
        test_probabilities,
    )

    # 8. Calculate the two required baseline costs.
    nobody_predictions = [0] * len(y_test)
    everybody_predictions = [1] * len(y_test)

    nobody_cost = calculate_cost(y_test, nobody_predictions)
    everybody_cost = calculate_cost(y_test, everybody_predictions)

    cost_comparison = {
        "nobody_contacted": nobody_cost,
        "everybody_contacted": everybody_cost,
        "trained_model": model_metrics["cost"],
    }

    model_beats_both = (
        model_metrics["cost"] < nobody_cost and model_metrics["cost"] < everybody_cost
    )

    print("\nTest-set model metrics:")
    for metric, value in model_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nTest-set cost comparison:")
    for name, cost in cost_comparison.items():
        print(f"{name}: ${cost:.2f}")

    print("Model beats both baselines:", model_beats_both)

    # 9. Save the complete fitted pipeline.
    model_path = MODELS_DIR / "no_show_pipeline.joblib"
    joblib.dump(best_model, model_path)

    # 10. Save model configuration and evaluation information.
    model_info = {
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "threshold": best_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "test_results": model_metrics,
        "baseline_costs": cost_comparison,
        "model_beats_both_baselines": model_beats_both,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    info_path = MODELS_DIR / "model_info.json"

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=4)

    test_results = {
        "model_metrics": model_metrics,
        "baseline_costs": cost_comparison,
        "model_beats_both_baselines": model_beats_both,
    }

    with (REPORTS_DIR / "test_results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(test_results, file, indent=4)

    print("\nSaved files:")
    print(model_path)
    print(info_path)
    print(REPORTS_DIR / "model_comparison.csv")
    print(REPORTS_DIR / "threshold_search.csv")
    print(REPORTS_DIR / "test_results.json")


if __name__ == "__main__":
    main()
