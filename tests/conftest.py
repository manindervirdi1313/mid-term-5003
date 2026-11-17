"""Small shared examples."""

import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor


@pytest.fixture
def booking_values():
    """A fictional, normal appointment we can change for different tests."""
    return {
        "lead_days": 10, "age": 40, "prior_no_shows": 0, "hour": 11,
        "weekday": "wed", "booking": "phone", "insurance": "private",
        "treatment": "physio", "first_visit": 0, "distance_km": 8.0,
    }


@pytest.fixture(scope="session")
def predictor():
    """Use the saved pipeline from training, not an invented fake classifier."""
    return NoShowPredictor()
