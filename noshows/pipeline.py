"""Preprocessing and model pipeline definitions."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 11  # One seed makes our results repeatable.
TEST_FRACTION = 0.25
TARGET = "no_show"
EXCLUDED_COLUMNS = ["appointment_id", "fee_charged", TARGET]

NUMERIC_FEATURES = [
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
]

CATEGORICAL_FEATURES = [
    "weekday",
    "booking",
    "insurance",
    "treatment",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "no_show"


def split_appointments(frame):
    """Keep 25% of the rows locked away for one final honest evaluation."""
    return train_test_split(
        frame[FEATURE_COLUMNS],
        frame[TARGET],
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=frame[TARGET],  # Both groups keep similar no-show percentages.
    )


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing steps for numeric and categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model, numeric_features=None):
    """Impute/scale numbers and one-hot encode choices before *any* classifier."""
    numbers = NUMERIC_FEATURES if numeric_features is None else numeric_features
    number_steps = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    category_steps = Pipeline(
        [
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            # An unfamiliar value cannot crash the encoder; Appointment blocks it.
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numbers", number_steps, numbers),
            ("categories", category_steps, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("prepare", transform), ("classifier", model)])


def candidate_models():
    """A guess-only baseline and four genuinely different model choices."""
    return {
        "Dummy (always most common)": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Balanced logistic regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "Small decision tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=SEED
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=1
        ),
    }
