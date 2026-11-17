"""Exercise the real page callback without opening a browser."""

from app import HIGH_RISK, LOW_RISK, check_booking


def test_page_shows_probability_for_good_booking():
    probability, action, explanation = check_booking(*LOW_RISK)
    assert "predicted no-show chance" in probability
    assert "No call needed" in action
    assert "training-data" not in explanation


def test_page_riskier_example_suggests_calling():
    probability, action, _ = check_booking(*HIGH_RISK)
    assert "predicted no-show chance" in probability
    assert "Call the patient" in action


def test_page_says_which_number_is_wrong():
    invalid = [121, *LOW_RISK[1:]]
    _, action, message = check_booking(*invalid)
    assert "lead_days" in message
    assert "No decision available" in action


def test_page_warns_but_accepts_missing_distance():
    blank_distance = [*LOW_RISK[:-1], None]
    probability, action, message = check_booking(*blank_distance)
    assert "predicted no-show chance" in probability
    assert "No call needed" in action
    assert "Distance was left blank" in message


def test_page_rejects_unknown_booking_choice():
    invalid = LOW_RISK.copy()
    invalid[5] = "carrier_pigeon"
    _, _, message = check_booking(*invalid)
    assert "booking" in message
