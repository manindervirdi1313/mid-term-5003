"""Tests for the saved no-show predictor."""

import json

import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor


@pytest.fixture
def predictor():
    """Load the trained predictor for testing."""

    return NoShowPredictor()


@pytest.fixture
def valid_appointment():
    """Provide a valid appointment."""

    return Appointment(
        lead_days=10,
        age=35,
        prior_no_shows=0,
        hour=10,
        weekday="mon",
        booking="online",
        insurance="private",
        treatment="physio",
        first_visit=0,
        distance_km=15.8,
    )


def test_predictor_loads_saved_model(predictor):
    """Predictor should load the saved model."""

    assert predictor.model is not None


def test_predictor_loads_threshold(predictor):
    """Threshold should match model_info.json."""

    with predictor.info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)

    assert predictor.threshold == info["threshold"]


def test_prediction_returns_probability(predictor, valid_appointment):
    """Prediction should return a probability between zero and one."""

    result = predictor.predict(valid_appointment)

    assert 0.0 <= result["no_show_probability"] <= 1.0


def test_prediction_contains_contact_decision(predictor, valid_appointment):
    """Prediction should include a contact decision and action."""

    result = predictor.predict(valid_appointment)

    assert isinstance(result["should_contact"], bool)
    assert result["action"] in {
        "Call the patient",
        "No call needed",
    }


def test_prediction_decision_matches_threshold(
    predictor,
    valid_appointment,
):
    """Contact decision should follow the configured threshold."""

    result = predictor.predict(valid_appointment)

    expected = result["no_show_probability"] >= predictor.threshold

    assert result["should_contact"] == expected


def test_predictor_has_readable_representation(predictor):
    """Predictor should provide a readable representation."""

    assert "NoShowPredictor" in repr(predictor)
    assert "threshold=" in repr(predictor)
