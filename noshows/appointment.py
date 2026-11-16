"""Appointment data validation and conversion."""

from dataclasses import dataclass

import pandas as pd

from noshows.pipeline import FEATURE_COLUMNS


@dataclass(frozen=True)
class Appointment:
    """Represent one appointment using booking-time information."""

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

    def __post_init__(self) -> None:
        """Validate appointment fields."""

        if not isinstance(self.lead_days, int) or not 0 <= self.lead_days <= 60:
            raise ValueError("lead_days must be an integer between 0 and 60")

        if not isinstance(self.age, int) or not 16 <= self.age <= 92:
            raise ValueError("age must be an integer between 16 and 92")

        if (
            not isinstance(self.prior_no_shows, int)
            or not 0 <= self.prior_no_shows <= 6
        ):
            raise ValueError("prior_no_shows must be an integer between 0 and 6")

        if not isinstance(self.hour, int) or not 8 <= self.hour <= 17:
            raise ValueError("hour must be an integer between 8 and 17")

        if self.weekday not in {"mon", "tue", "wed", "thu", "fri"}:
            raise ValueError("weekday must be mon, tue, wed, thu, or fri")

        if self.booking not in {"online", "phone", "front_desk"}:
            raise ValueError("booking must be online, phone, or front_desk")

        if self.insurance not in {"private", "provincial", "self_pay"}:
            raise ValueError("insurance must be private, provincial, or self_pay")

        if self.treatment not in {"physio", "massage", "acupuncture"}:
            raise ValueError("treatment must be physio, massage, or acupuncture")

        if self.first_visit not in {0, 1}:
            raise ValueError("first_visit must be 0 or 1")

        if self.distance_km is not None:
            if not 0.5 <= self.distance_km <= 60:
                raise ValueError("distance_km must be between 0.5 and 60")

    def to_frame(self) -> pd.DataFrame:
        """Convert the appointment into one model-ready dataframe row."""

        values = {
            "lead_days": self.lead_days,
            "age": self.age,
            "prior_no_shows": self.prior_no_shows,
            "hour": self.hour,
            "weekday": self.weekday,
            "booking": self.booking,
            "insurance": self.insurance,
            "treatment": self.treatment,
            "first_visit": self.first_visit,
            "distance_km": self.distance_km,
        }

        return pd.DataFrame([values], columns=FEATURE_COLUMNS)
