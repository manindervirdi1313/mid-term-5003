"""Maple Physio: a simple, original Gradio webpage for the front desk.

Only Appointment checks inputs; only NoShowPredictor makes predictions.
No private patient information is stored. All demo appointments are invented.
"""

import gradio as gr

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor

try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU
    def _free_zerogpu_placeholder():
        """Meet the ZeroGPU Space startup check (no GPU is actually needed)."""
        return None


try:
    predictor = NoShowPredictor()
    load_error = None
except (FileNotFoundError, ValueError, OSError) as error:
    # Keep the page visible even when a new student has not trained the model.
    predictor = None
    load_error = str(error)


def check_booking(
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
    """Connect the front desk form to our two tested Python classes."""
    if predictor is None:
        return (
            "**Model not ready.** Run `python -m noshows.train` first.",
            "No decision available",
            load_error or "The saved model could not be loaded.",
        )
    try:
        appointment = Appointment(
            lead_days=lead_days,
            age=age,
            prior_no_shows=prior_no_shows,
            hour=hour,
            weekday=weekday,
            booking=booking,
            insurance=insurance,
            treatment=treatment,
            first_visit=first_visit,
            distance_km=distance_km,
        )
        result = predictor.predict(appointment)
    except (ValueError, TypeError) as error:
        return ("**Please fix your booking.**", "No decision available", f"❌ {error}")

    percent = 100 * result["no_show_probability"]
    cutoff = 100 * predictor.threshold
    warning = (
        "⚠️ **Distance was left blank.** We used the typical distance from the "
        "training data, so this estimate may be less precise."
        if appointment.distance_km is None
        else "✅ All appointment fields were provided."
    )
    return (
        f"## {percent:.1f}% predicted no-show chance\n"
        f"A call is suggested at **{cutoff:.0f}%** risk or higher.",
        f"### {'📞 ' if result['should_contact'] else '✅ '}{result['action']}",
        f"{result['explanation']}\n\n{warning}",
    )


# These example values are fictional. They deliberately contrast low/high risk.
LOW_RISK = [2, 65, 0, 11, "wed", "front_desk", "provincial", "acupuncture", 0, 2.0]
HIGH_RISK = [12, 40, 1, 10, "mon", "online", "self_pay", "physio", 1, 12.0]

with gr.Blocks(title="Maple Physio | No-Show Decision Helper") as demo:
    gr.Markdown(
        """# 🏥 Maple Physio | Appointment Check
        **One appointment. One estimate. One clear call decision.**

        Enter **invented demo bookings** below. A phone call costs **$8**;
        an uncalled missed visit costs **$35**. Our recommendation compares those costs.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1 · Appointment details")
            with gr.Row():
                lead_days = gr.Number(
                    label="Days until visit (0–120)", value=10, precision=0
                )
                age = gr.Number(label="Age (12–110)", value=35, precision=0)
            with gr.Row():
                prior_no_shows = gr.Number(
                    label="Previous missed visits (0–30)", value=0, precision=0
                )
                hour = gr.Number(
                    label="Start hour (8–17, 24-hour clock)", value=10, precision=0
                )
            with gr.Row():
                weekday = gr.Dropdown(
                    label="Weekday",
                    choices=Appointment.CHOICES["weekday"],
                    value="wed",
                )
                booking = gr.Dropdown(
                    label="How was it booked?",
                    choices=Appointment.CHOICES["booking"],
                    value="phone",
                )
            with gr.Row():
                insurance = gr.Dropdown(
                    label="Insurance",
                    choices=Appointment.CHOICES["insurance"],
                    value="private",
                )
                treatment = gr.Dropdown(
                    label="Treatment",
                    choices=Appointment.CHOICES["treatment"],
                    value="physio",
                )
            with gr.Row():
                first_visit = gr.Radio(
                    label="First visit? (0 = No, 1 = Yes)", choices=[0, 1], value=0
                )
                distance_km = gr.Number(
                    label="Distance in km (0.1–150; blank is allowed)", value=8.0
                )
            inputs = [
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
            ]
            with gr.Row():
                check = gr.Button("Check appointment", variant="primary")
                low = gr.Button("Try safer example")
                high = gr.Button("Try riskier example")
        with gr.Column(scale=1):
            gr.Markdown("### 2 · Results")
            risk = gr.Markdown("**Enter a booking and select Check appointment.**")
            action = gr.Markdown("Recommendation appears here.")
            details = gr.Markdown("No prediction has been made yet.")
            gr.Markdown(
                """---
                **What does the percentage mean?** 30% means the model expects
                about 30 misses among 100 similar invented bookings.

                **Remember:** This is a class exercise using made-up data, not
                a medical decision or a replacement for a receptionist.
                """
            )
    check.click(check_booking, inputs=inputs, outputs=[risk, action, details])
    low.click(lambda: LOW_RISK, outputs=inputs)
    high.click(lambda: HIGH_RISK, outputs=inputs)


if __name__ == "__main__":
    demo.launch()
