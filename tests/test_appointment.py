"""Appointment must protect the model from invalid receptionist input."""

from dataclasses import FrozenInstanceError

import pytest

from noshows.appointment import Appointment
from noshows.pipeline import FEATURE_COLUMNS


def test_valid_booking_converts_to_one_row(booking_values):
    appointment = Appointment(**booking_values)
    result = appointment.to_frame()
    assert result.shape == (1, len(FEATURE_COLUMNS))
    assert list(result.columns) == FEATURE_COLUMNS


def test_allowed_missing_distance_is_nan(booking_values):
    appointment = Appointment(**{**booking_values, "distance_km": None})
    assert appointment.to_frame()["distance_km"].isna().all()


def test_booking_is_frozen(booking_values):
    appointment = Appointment(**booking_values)
    with pytest.raises(FrozenInstanceError):
        appointment.age = 28


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("lead_days", -1), ("lead_days", 121), ("lead_days", 10.5),
        ("age", 11), ("age", 111), ("prior_no_shows", -1),
        ("prior_no_shows", 31), ("hour", 7), ("hour", 18),
        ("first_visit", 2), ("first_visit", -1),
        ("distance_km", 0), ("distance_km", 150.1),
        ("distance_km", float("nan")), ("age", None),
        ("first_visit", True),
    ],
)
def test_impossible_numbers_name_the_field(booking_values, field, bad):
    with pytest.raises(ValueError, match=field):
        Appointment(**{**booking_values, field: bad})


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("weekday", "sat"), ("booking", "fax"),
        ("insurance", "none"), ("treatment", "xray"),
    ],
)
def test_unseen_categories_name_the_field(booking_values, field, bad):
    with pytest.raises(ValueError, match=field):
        Appointment(**{**booking_values, field: bad})


def test_wider_assignments_limits_work(booking_values):
    appointment = Appointment(**{
        **booking_values, "lead_days": 120, "age": 110,
        "prior_no_shows": 30, "distance_km": 150.0,
    })
    assert appointment.lead_days == 120


def test_whole_number_from_gradio_is_accepted(booking_values):
    appointment = Appointment(**{**booking_values, "age": 40.0})
    assert appointment.age == 40
