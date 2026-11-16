"""Tests for Appointment validation and conversion."""

import pytest

from noshows.appointment import Appointment
from noshows.pipeline import FEATURE_COLUMNS


def make_valid_appointment(**overrides):
    """Create a valid appointment for testing."""

    values = {
        "lead_days": 10,
        "age": 35,
        "prior_no_shows": 0,
        "hour": 10,
        "weekday": "mon",
        "booking": "online",
        "insurance": "private",
        "treatment": "physio",
        "first_visit": 0,
        "distance_km": 15.8,
    }

    values.update(overrides)

    return Appointment(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", 121),
        ("age", 11),
        ("prior_no_shows", -1),
        ("hour", 18),
        ("weekday", "saturday"),
        ("booking", "unknown"),
        ("insurance", "unknown"),
        ("treatment", "unknown"),
        ("first_visit", 2),
        ("distance_km", 151.0),
    ],
)
def test_invalid_fields_raise_value_error(field, value):
    """Invalid appointment values should name the field."""

    with pytest.raises(ValueError, match=field):
        make_valid_appointment(**{field: value})


def test_valid_appointment_converts_to_one_row():
    """Appointment should become one row with all model features."""

    appointment = make_valid_appointment()

    frame = appointment.to_frame()

    assert frame.shape == (1, len(FEATURE_COLUMNS))
    assert list(frame.columns) == FEATURE_COLUMNS


def test_missing_distance_is_allowed():
    """Missing distance should be accepted for later imputation."""

    appointment = make_valid_appointment(distance_km=None)

    assert appointment.to_frame()["distance_km"].isna().all()


def test_appointment_is_immutable():
    """Frozen appointment fields should not be reassigned."""

    appointment = make_valid_appointment()

    with pytest.raises(AttributeError):
        appointment.age = 40
