"""A fitted model is loaded once and must follow the saved threshold."""

import json
from pathlib import Path
from unittest.mock import Mock

import joblib
import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor


def test_probability_is_between_zero_and_one(predictor, booking_values):
    result = predictor.predict(Appointment(**booking_values))
    assert 0 <= result["no_show_probability"] <= 1


def test_riskier_booking_scores_higher(predictor, booking_values):
    safe = Appointment(**{
        **booking_values, "lead_days": 2, "age": 65,
        "prior_no_shows": 0, "weekday": "wed", "booking": "front_desk",
        "insurance": "provincial", "first_visit": 0, "distance_km": 2,
    })
    risky = Appointment(**{
        **booking_values, "lead_days": 55, "age": 22,
        "prior_no_shows": 5, "weekday": "mon", "booking": "online",
        "insurance": "self_pay", "first_visit": 1, "distance_km": 55,
    })
    assert (
        predictor.predict(risky)["no_show_probability"]
        > predictor.predict(safe)["no_show_probability"]
    )


def test_decision_matches_model_threshold(predictor, booking_values):
    result = predictor.predict(Appointment(**booking_values))
    assert result["should_contact"] == (
        result["no_show_probability"] >= predictor.threshold
    )


def test_threshold_came_from_json(predictor):
    with predictor.info_path.open(encoding="utf-8") as handle:
        assert predictor.threshold == json.load(handle)["threshold"]


def test_predictor_representation_is_readable(predictor):
    assert "NoShowPredictor(" in repr(predictor)
    assert "threshold=" in repr(predictor)


def test_predictor_rejects_other_objects(predictor):
    with pytest.raises(TypeError, match="Appointment"):
        predictor.predict({"lead_days": 10})


def test_joblib_model_is_loaded_exactly_once(monkeypatch, tmp_path, predictor):
    """Count load calls: a fresh object may read the file only one time."""
    from noshows import predictor as module

    source_info = Path(predictor.info_path)
    calls = Mock(wraps=joblib.load)
    monkeypatch.setattr(module.joblib, "load", calls)
    fresh = NoShowPredictor(model_path=predictor.model_path, info_path=source_info)
    assert fresh.model is not None
    assert calls.call_count == 1


def test_prediction_action_is_human_friendly(predictor, booking_values):
    result = predictor.predict(Appointment(**booking_values))
    assert result["action"] in {"Call the patient", "No call needed"}
    assert result["explanation"]
