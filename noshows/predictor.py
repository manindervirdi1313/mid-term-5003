"""Load the trained model and make appointment predictions."""

import json
from pathlib import Path

import joblib

from noshows.appointment import Appointment


class NoShowPredictor:
    """Make no-show predictions using a saved model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        info_path: str | Path | None = None,
    ) -> None:
        """Load the saved pipeline and model information."""

        project_root = Path(__file__).resolve().parent.parent

        if model_path is None:
            model_path = project_root / "models" / "no_show_pipeline.joblib"

        if info_path is None:
            info_path = project_root / "models" / "model_info.json"

        self.model_path = Path(model_path)
        self.info_path = Path(info_path)

        self.model = joblib.load(self.model_path)

        with self.info_path.open("r", encoding="utf-8") as file:
            self.model_info = json.load(file)

        self.threshold = float(self.model_info["threshold"])

    def predict(self, appointment: Appointment) -> dict:
        """Return the no-show probability and contact decision."""

        frame = appointment.to_frame()

        probability = float(self.model.predict_proba(frame)[0, 1])

        should_contact = probability >= self.threshold

        return {
            "no_show_probability": probability,
            "should_contact": should_contact,
            "action": "Contact patient" if should_contact else "No contact needed",
        }

    def __repr__(self) -> str:
        """Return a readable predictor description."""

        return (
            f"NoShowPredictor(threshold={self.threshold:.2f}, "
            f"model_path='{self.model_path.name}')"
        )
