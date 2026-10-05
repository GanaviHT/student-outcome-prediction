"""
Unit tests for the leakage-safety rules. Run from the project root with:

    python -m pytest tests -q

(They use the real data.csv and a fast Logistic Regression, so they take a few seconds.)
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_loader
from src.models import get_models
from src.preprocessing import (build_pipeline, build_preprocessor,
                               make_train_test_split, split_features_target)
from src.predict import predict_students


@lru_cache(maxsize=1)
def _data():
    df = data_loader.load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    return df, X, y, X_train, X_test, y_train, y_test


def test_dataset_shape_and_columns():
    df, *_ = _data()
    assert df.shape == (4424, 37)
    data_loader.check_columns(df)


def test_prohibited_columns_never_used():
    _, X, *_ = _data()
    assert not set(X.columns) & set(config.PROHIBITED_COLUMNS)
    assert not set(config.FEATURE_COLUMNS) & set(config.PROHIBITED_COLUMNS)
    assert len(config.PROHIBITED_COLUMNS) == 12
    assert X.shape[1] == 24
    assert config.TARGET not in X.columns


def test_feature_groups_are_disjoint_and_complete():
    groups = [config.NUMERIC_FEATURES, config.BINARY_FEATURES, config.CATEGORICAL_FEATURES]
    flat = sum(groups, [])
    assert len(flat) == len(set(flat)) == 24


def test_split_is_disjoint_and_stratified():
    _, X, y, X_train, X_test, y_train, y_test = _data()
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - config.TEST_SIZE) < 0.01
    for cls in config.CLASS_ORDER:
        assert abs((y_train == cls).mean() - (y_test == cls).mean()) < 0.01


def test_split_is_reproducible():
    _, X, y, X_train, *_ = _data()
    X_train2, *_ = make_train_test_split(X, y)
    assert list(X_train.index) == list(X_train2.index)


def test_preprocessor_learns_from_training_data_only():
    _, X, _, X_train, X_test, *_ = _data()
    pre = build_preprocessor().fit(X_train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, X_train[config.NUMERIC_FEATURES].mean().values)
    # ...and it is NOT the mean of the test set or of all students
    assert not np.allclose(scaler.mean_, X_test[config.NUMERIC_FEATURES].mean().values)
    assert not np.allclose(scaler.mean_, X[config.NUMERIC_FEATURES].mean().values)


def test_transform_does_not_change_fitted_statistics():
    _, _, _, X_train, X_test, *_ = _data()
    pre = build_preprocessor().fit(X_train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    before = scaler.mean_.copy()
    pre.transform(X_test)
    np.testing.assert_array_equal(before, scaler.mean_)


def test_pipeline_predicts_valid_labels_and_probabilities():
    _, _, _, X_train, X_test, y_train, _ = _data()
    pipe = build_pipeline(get_models()["Logistic Regression (balanced)"]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    assert len(pred) == len(X_test)
    assert set(pred) <= set(config.CLASS_ORDER)
    proba = pipe.predict_proba(X_test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_unseen_category_code_does_not_crash():
    _, _, _, X_train, X_test, y_train, _ = _data()
    pipe = build_pipeline(get_models()["Logistic Regression (balanced)"]).fit(X_train, y_train)
    weird = X_test.head(3).copy()
    weird["Course"] = 99999          # a code never seen in training
    assert len(pipe.predict(weird)) == 3


def test_predict_students_ignores_extra_columns_and_checks_missing():
    df, _, _, X_train, X_test, y_train, _ = _data()
    pipe = build_pipeline(get_models()["Logistic Regression (balanced)"]).fit(X_train, y_train)
    full_rows = df.loc[X_test.index[:3]]                  # still has semester columns + Target
    out = predict_students(pipe, full_rows)
    assert list(out.columns)[0] == "predicted_outcome" and len(out) == 3
    try:
        predict_students(pipe, X_test.drop(columns=["Course"]).head(2))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a missing column")


if __name__ == "__main__":       # lets you run:  python tests/test_pipeline.py
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("PASS", n)
    print(f"{len(names)} tests passed")
