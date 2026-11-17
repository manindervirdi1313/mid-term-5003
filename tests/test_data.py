"""Checks for the professor UNCHANGED invented dataset."""

import pandas as pd

from noshows.data import make_appointments
from noshows.pipeline import (
    CATEGORICAL_FEATURES, EXCLUDED_COLUMNS, FEATURE_COLUMNS,
    NUMERIC_FEATURES, SEED, TARGET, split_appointments,
)


def test_data_can_be_recreated():
    """Using the same seed should produce the exact same invented rows."""
    pd.testing.assert_frame_equal(make_appointments(), make_appointments())


def test_data_has_4000_rows_and_expected_columns():
    data = make_appointments()
    assert len(data) == 4000
    assert len(data.columns) == 13


def test_target_only_has_zero_or_one():
    assert set(make_appointments()[TARGET].unique()) == {0, 1}


def test_some_distances_are_missing():
    assert make_appointments()["distance_km"].isna().any()


def test_feature_lists_keep_out_leaks():
    assert set(EXCLUDED_COLUMNS) == {"appointment_id", "fee_charged", TARGET}
    assert not set(EXCLUDED_COLUMNS).intersection(FEATURE_COLUMNS)
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_split_is_75_25():
    a, b, c, d = split_appointments(make_appointments())
    assert (len(a), len(b), len(c), len(d)) == (3000, 1000, 3000, 1000)
    assert not set(a.index).intersection(set(b.index))


def test_split_is_stratified():
    _, _, y_train, y_test = split_appointments(make_appointments())
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_repeats_with_one_seed():
    x_first, *_ = split_appointments(make_appointments(seed=SEED))
    x_second, *_ = split_appointments(make_appointments(seed=SEED))
    pd.testing.assert_frame_equal(x_first, x_second)
