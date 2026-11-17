"""Tests for costs and training-only threshold search."""

import numpy as np
import pytest

from noshows.evaluation import (
    THRESHOLD_GRID, calculate_cost, choose_threshold, make_calls,
    savings_interval,
)


def test_hand_worked_cost_is_51():
    """2 calls x $8 + 1 uncalled no-show x $35 = $51."""
    assert calculate_cost([1, 0, 1, 0], [0, 0, 1, 1]) == 51


def test_nobody_calls_costs_70_in_small_example():
    assert calculate_cost([1, 0, 1, 0], [0, 0, 0, 0]) == 70


def test_everybody_calls_costs_32_in_small_example():
    assert calculate_cost([1, 0, 1, 0], [1, 1, 1, 1]) == 32


def test_calling_nobody_is_cheapest_when_nobody_misses():
    assert calculate_cost([0, 0, 0], [0, 0, 0]) < calculate_cost(
        [0, 0, 0], [1, 1, 1]
    )


def test_calling_everybody_is_cheapest_when_everybody_misses():
    assert calculate_cost([1, 1, 1], [1, 1, 1]) < calculate_cost(
        [1, 1, 1], [0, 0, 0]
    )


def test_cost_rejects_different_lengths():
    with pytest.raises(ValueError, match="same shape"):
        calculate_cost([0, 1], [1])


@pytest.mark.parametrize(
    ("chance", "expected"),
    [(0.499999, 0), (0.5, 1), (0.500001, 1)],
)
def test_exact_threshold_boundary(chance, expected):
    """Just below / equal / just above the cutoff."""
    assert make_calls([chance], 0.5).item() == expected


def test_threshold_grid_has_all_91_choices():
    assert len(THRESHOLD_GRID) == 91
    assert THRESHOLD_GRID[0] == 0.05
    assert THRESHOLD_GRID[-1] == 0.95


def test_threshold_when_call_is_never_worth_it():
    """If calls cost $1,000 but missed visits cost $1, choose no calls."""
    threshold, table = choose_threshold(
        [1, 0, 1], [0.1, 0.4, 0.6], call_cost=1000, miss_cost=1
    )
    assert threshold > 0.6
    assert int(table.loc[table.threshold == threshold, "calls"].iloc[0]) == 0


def test_threshold_when_a_miss_is_very_costly():
    """If missing is $1,000 but calling is $1, call everybody."""
    threshold, table = choose_threshold(
        [1, 1, 1], [0.1, 0.4, 0.6], call_cost=1, miss_cost=1000
    )
    assert threshold <= 0.1
    assert int(table.loc[table.threshold == threshold, "calls"].iloc[0]) == 3


def test_bootstrap_interval_returns_ordered_numbers():
    result = savings_interval(
        [1, 0, 0, 1], [1, 0, 0, 0], np.zeros(4), repeats=100
    )
    assert len(result) == 2 and result[0] <= result[1]
