"""The cleaning conveyor belt must behave the same for every model."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from noshows.data import make_appointments
from noshows.pipeline import build_pipeline, candidate_models, split_appointments


def test_pipeline_wraps_the_model():
    pipeline = build_pipeline(DummyClassifier())
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["prepare", "classifier"]


def test_pipeline_can_fill_in_missing_distance():
    x_train, _, y_train, _ = split_appointments(make_appointments())
    model = build_pipeline(DummyClassifier()).fit(x_train, y_train)
    sample = x_train.iloc[[0]].copy()
    sample["distance_km"] = np.nan
    assert model.predict_proba(sample).shape == (1, 2)


def test_each_candidate_uses_same_pipeline():
    assert len(candidate_models()) >= 4
    for model in candidate_models().values():
        assert list(build_pipeline(model).named_steps) == ["prepare", "classifier"]


def test_encoder_ignores_unseen_choice_during_preprocessing():
    x_train, _, y_train, _ = split_appointments(make_appointments())
    model = build_pipeline(DummyClassifier()).fit(x_train, y_train)
    sample = x_train.iloc[[0]].copy()
    sample["weekday"] = "not_a_real_day"
    # This is why Appointment must block invalid categories before predicting.
    assert model.predict_proba(sample).shape == (1, 2)
