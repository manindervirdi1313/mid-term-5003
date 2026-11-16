"""Tests for evaluation and cost calculations."""

import numpy as np
import pytest

from noshows.evaluation import (
    calculate_cost,
    get_threshold_predictions,
)


def test_cost_calculation():
    """Cost should include flagged appointments and missed no-shows."""

    actual = [1, 0, 1, 0]
    predicted = [0, 0, 1, 1]

    assert calculate_cost(actual, predicted) == 51.0


def test_cost_when_nobody_is_contacted():
    """Nobody contacted means every no-show incurs a $35 cost."""

    actual = [1, 0, 1, 0]

    assert calculate_cost(actual, [0, 0, 0, 0]) == 70.0


def test_cost_when_everybody_is_contacted():
    """Everybody contacted means every appointment costs $8."""

    actual = [1, 0, 1, 0]

    assert calculate_cost(actual, [1, 1, 1, 1]) == 32.0


@pytest.mark.parametrize(
    ("probabilities", "threshold", "expected"),
    [
        ([0.1, 0.5, 0.9], 0.5, [0, 1, 1]),
        ([0.1, 0.5, 0.9], 0.95, [0, 0, 0]),
        ([0.1, 0.5, 0.9], 0.05, [1, 1, 1]),
    ],
)
def test_threshold_predictions(probabilities, threshold, expected):
    """Threshold should correctly convert probabilities to decisions."""

    predictions = get_threshold_predictions(
        probabilities,
        threshold,
    )

    assert predictions.tolist() == expected


def test_threshold_predictions_return_binary_values():
    """Predictions should contain only zero and one."""

    predictions = get_threshold_predictions(
        np.array([0.2, 0.4, 0.6, 0.8]),
        0.5,
    )

    assert set(predictions.tolist()) <= {0, 1}
