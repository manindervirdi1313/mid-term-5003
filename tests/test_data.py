"""Tests for the appointment dataset."""

import pandas as pd

from noshows.data import make_appointments


def test_dataset_has_expected_shape():
    data = make_appointments()
    assert data.shape == (4000, 13)


def test_dataset_is_reproducible():
    first = make_appointments(seed=11)
    second = make_appointments(seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_data():
    first = make_appointments(seed=11)
    second = make_appointments(seed=12)
    assert not first.equals(second)


def test_target_contains_only_binary_values():
    data = make_appointments()
    assert set(data["no_show"].unique()) == {0, 1}


def test_dataset_contains_missing_distance_values():
    data = make_appointments()
    assert data["distance_km"].isna().any()
