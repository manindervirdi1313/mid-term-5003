"""Gradio application for physiotherapy no-show prediction."""

import gradio as gr

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor

predictor = NoShowPredictor()


def predict_no_show(
    lead_days,
    age,
    prior_no_shows,
    hour,
    weekday,
    booking,
    insurance,
    treatment,
    first_visit,
    distance_km,
):
    """Validate appointment details and predict no-show risk."""

    try:
        distance_warning = ""

        if distance_km is None:
            distance_warning = "⚠️ Distance is missing. Median imputation will be used."

        appointment = Appointment(
            lead_days=int(lead_days),
            age=int(age),
            prior_no_shows=int(prior_no_shows),
            hour=int(hour),
            weekday=weekday,
            booking=booking,
            insurance=insurance,
            treatment=treatment,
            first_visit=int(first_visit),
            distance_km=(None if distance_km is None else float(distance_km)),
        )

        result = predictor.predict(appointment)

        probability = result["no_show_probability"] * 100

        output = f"**Estimated no-show probability: {probability:.1f}%**"

        return (
            f"{distance_warning}\n\n{output}",
            result["action"],
        )

    except (ValueError, TypeError) as error:
        return f"Input error: {error}", ""


with gr.Blocks(title="Maple Physio No-Show Predictor") as demo:
    gr.Markdown(
        """
        # Maple Physio Appointment Risk Predictor

        Enter appointment details to estimate the probability of a
        patient missing their appointment.

        The model uses information available at booking time.
        """
    )

    with gr.Row():
        lead_days = gr.Slider(
            minimum=0,
            maximum=60,
            value=10,
            step=1,
            label="Days between booking and appointment",
        )

        age = gr.Slider(
            minimum=16,
            maximum=92,
            value=35,
            step=1,
            label="Patient age",
        )

    with gr.Row():
        prior_no_shows = gr.Slider(
            minimum=0,
            maximum=6,
            value=0,
            step=1,
            label="Previous no-shows",
        )

        hour = gr.Slider(
            minimum=8,
            maximum=17,
            value=10,
            step=1,
            label="Appointment hour",
        )

    with gr.Row():
        weekday = gr.Dropdown(
            choices=["mon", "tue", "wed", "thu", "fri"],
            value="mon",
            label="Weekday",
        )

        booking = gr.Dropdown(
            choices=["online", "phone", "front_desk"],
            value="online",
            label="Booking method",
        )

    with gr.Row():
        insurance = gr.Dropdown(
            choices=["private", "provincial", "self_pay"],
            value="private",
            label="Insurance type",
        )

        treatment = gr.Dropdown(
            choices=["physio", "massage", "acupuncture"],
            value="physio",
            label="Treatment type",
        )

    first_visit = gr.Radio(
        choices=[0, 1],
        value=0,
        label="Is this the patient's first visit?",
    )

    distance_km = gr.Number(
        value=15.8,
        label="Distance from clinic (km)",
        minimum=0.5,
        maximum=60,
    )

    predict_button = gr.Button("Predict No-Show Risk", variant="primary")

    probability_output = gr.Markdown(label="Prediction")
    action_output = gr.Markdown(label="Recommended Action")

    predict_button.click(
        fn=predict_no_show,
        inputs=[
            lead_days,
            age,
            prior_no_shows,
            hour,
            weekday,
            booking,
            insurance,
            treatment,
            first_visit,
            distance_km,
        ],
        outputs=[probability_output, action_output],
    )


if __name__ == "__main__":
    demo.launch()
