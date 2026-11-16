"""Load the trained model and make appointment predictions."""

import json
from pathlib import Path

import joblib

from noshows.appointment import Appointment
from noshows.pipeline import FEATURE_COLUMNS


class NoShowPredictor:
    """Make no-show predictions using a saved model."""

    def __init__(self, model_path=None, info_path=None):
        project_root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path or project_root / "models" / "no_show.joblib")
        self.info_path = Path(info_path or project_root / "models" / "model_info.json")

        # Load ONCE per instance. Never load again when a user presses Predict.
        self.model = joblib.load(self.model_path)
        with self.info_path.open(encoding="utf-8") as handle:
            self.model_info = json.load(handle)
        if self.model_info["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("Saved model feature_columns do not match this code")
        self.threshold = float(self.model_info["threshold"])
        if not 0 < self.threshold < 1:
            raise ValueError("Saved model threshold must be between 0 and 1")

    def predict(self, appointment):
        """Give the no-show chance, yes/no call decision, and friendly advice."""
        if not isinstance(appointment, Appointment):
            raise TypeError("appointment must be an Appointment object")
        probability = float(self.model.predict_proba(appointment.to_frame())[0, 1])
        call = probability >= self.threshold
        return {
            "no_show_probability": probability,
            "should_contact": bool(call),
            "action": "Call the patient" if call else "No call needed",
            "explanation": (
                "The predicted risk meets the clinic's cost-saving cutoff."
                if call
                else "The predicted risk is below the clinic's cost-saving cutoff."
            ),
        }

    def __repr__(self):
        return (
            f"NoShowPredictor(threshold={self.threshold:.2f}, "
            f"model='{self.model_path.name}')"
        )
