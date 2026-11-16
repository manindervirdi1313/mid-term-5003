"""Appointment data validation and conversion.
One booking with strict front-desk checks before it reaches the model."""

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

import pandas as pd

from noshows.pipeline import FEATURE_COLUMNS


@dataclass(frozen=True)
class Appointment:
    """An immutable (unchangeable) appointment with valid booking-time inputs."""

    # Use these SAME choices in the webpage and in the checks below.
    CHOICES: ClassVar[dict[str, tuple[str, ...]]] = {
        "weekday": ("mon", "tue", "wed", "thu", "fri"),
        "booking": ("online", "phone", "front_desk"),
        "insurance": ("private", "provincial", "self_pay"),
        "treatment": ("physio", "massage", "acupuncture"),
    }
    INTEGER_LIMITS: ClassVar[dict[str, tuple[int, int]]] = {
        "lead_days": (0, 120),
        "age": (12, 110),
        "prior_no_shows": (0, 30),
        "hour": (8, 17),
        "first_visit": (0, 1),
    }
    DISTANCE_LIMITS: ClassVar[tuple[float, float]] = (0.1, 150.0)

    lead_days: int
    age: int
    prior_no_shows: int
    hour: int
    weekday: str
    booking: str
    insurance: str
    treatment: str
    first_visit: int
    distance_km: float | None

    def __post_init__(self):
        """Raise ValueError with the exact field name for each bad input."""
        for field, (minimum, maximum) in self.INTEGER_LIMITS.items():
            value = getattr(self, field)
            # A Gradio Number may supply 12.0; accept it, but not 12.5.
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not float(value).is_integer()
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{field} must be a whole number from {minimum} to {maximum}"
                )
            # Frozen fields can still be normalized while the object is created.
            object.__setattr__(self, field, int(value))

        for field, choices in self.CHOICES.items():
            if getattr(self, field) not in choices:
                raise ValueError(f"{field} must be one of: {', '.join(choices)}")

        distance = self.distance_km
        if distance is not None:
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not isfinite(distance)
                or not self.DISTANCE_LIMITS[0] <= distance <= self.DISTANCE_LIMITS[1]
            ):
                raise ValueError("distance_km must be between 0.1 and 150, or blank")
            object.__setattr__(self, "distance_km", float(distance))

    def to_frame(self):
        """Put values in EXACTLY the order of the model's feature columns."""
        return pd.DataFrame(
            [{field: getattr(self, field) for field in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )
