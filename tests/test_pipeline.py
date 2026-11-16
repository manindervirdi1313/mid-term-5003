"""Tests for the data split and model pipelines."""

import pytest

from noshows.data import make_appointments
from noshows.pipeline import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    build_pipeline,
    candidate_models,
    split_appointments,
)


@pytest.fixture(scope="module")
def appointments():
    """Build the dataset once and share it across tests."""

    return make_appointments()


def test_split_keeps_one_quarter_for_testing(appointments):
    """The locked test set should hold 25% of the rows."""

    x_train, x_test, y_train, y_test = split_appointments(appointments)

    assert len(x_test) == 1000
    assert len(x_train) == 3000
    assert len(y_test) == len(x_test)
    assert len(y_train) == len(x_train)


def test_split_is_repeatable(appointments):
    """The same seed should always give the same test rows."""

    first = split_appointments(appointments)[1]
    second = split_appointments(appointments)[1]

    assert first.index.equals(second.index)


def test_split_keeps_similar_no_show_rates(appointments):
    """Stratifying should keep the no-show rate close in both groups."""

    _, _, y_train, y_test = split_appointments(appointments)

    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_excludes_id_fee_and_target(appointments):
    """Leaky or useless columns must never reach the model."""

    x_train, x_test, _, _ = split_appointments(appointments)

    for column in EXCLUDED_COLUMNS:
        assert column not in x_train.columns
        assert column not in x_test.columns
    assert list(x_train.columns) == FEATURE_COLUMNS


def test_candidate_models_include_dummy_baseline():
    """A guess-only baseline plus four real models should be compared."""

    models = candidate_models()

    assert len(models) == 5
    assert "Dummy (always most common)" in models


@pytest.mark.parametrize("name", list(candidate_models()))
def test_every_pipeline_trains_with_missing_distance(appointments, name):
    """Each pipeline should handle blank distances and return probabilities."""

    x_train, x_test, y_train, _ = split_appointments(appointments)
    assert x_train["distance_km"].isna().any()

    pipeline = build_pipeline(candidate_models()[name])
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    assert len(probabilities) == len(x_test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_pipeline_ignores_unseen_category(appointments):
    """An unfamiliar category value should not crash a prediction."""

    x_train, x_test, y_train, _ = split_appointments(appointments)
    pipeline = build_pipeline(candidate_models()["Logistic regression"])
    pipeline.fit(x_train, y_train)

    row = x_test.head(1).copy()
    row["weekday"] = "sat"

    assert pipeline.predict_proba(row).shape == (1, 2)
