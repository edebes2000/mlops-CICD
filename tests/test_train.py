"""
Educational Goal:
- Why this test exists in an MLOps system: Confirm train.py produces a fitted model artifact
- Responsibility (separation of concerns): Test training without depending on features.py
- Pipeline contract: Valid inputs return a fitted artifact, invalid inputs raise clear errors
"""

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.train import calibrate_pipeline, train_model


@pytest.fixture
def dummy_recipe_and_data():
    """
    Small deterministic dataset for fast tests.

    Notes
    - Keep enough rows so calibration cross-validation has both classes in each fold
    - Preprocessor is deliberately simple so this test stays isolated from features.py
    """
    X_train = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50],
            "dose": [10, 20, 30, 40, 50, 60],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1])

    preprocessor = ColumnTransformer(
        transformers=[("pass", "passthrough", ["age", "dose"])]
    )

    return X_train, y_train, preprocessor


def test_train_model_returns_pipeline_with_expected_steps(dummy_recipe_and_data):
    X_train, y_train, preprocessor = dummy_recipe_and_data

    pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type="classification",
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocess" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")


def test_train_model_ignores_calibration_keys_when_building_base_classifier(dummy_recipe_and_data):
    """
    Why this matters
    - Calibration settings live in config.yaml under training.classification
    - They must not be passed directly into LogisticRegression(...)
    """
    X_train, y_train, preprocessor = dummy_recipe_and_data

    pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type="classification",
        model_params={
            "model_type": "logistic_regression",
            "max_iter": 200,
            "solver": "liblinear",
            "random_state": 42,
            "class_weight": "balanced",
            "calibration_enabled": True,
            "calibration_method": "sigmoid",
            "calibration_cv": 3,
        },
    )

    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["model"].__class__.__name__ == "LogisticRegression"


def test_calibrate_pipeline_returns_calibrated_model(dummy_recipe_and_data):
    X_train, y_train, preprocessor = dummy_recipe_and_data

    base_pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type="classification",
        model_params={
            "model_type": "logistic_regression",
            "max_iter": 200,
            "solver": "liblinear",
            "random_state": 42,
        },
    )

    calibrated_model = calibrate_pipeline(
        pipeline=base_pipeline,
        X_train=X_train,
        y_train=y_train,
        method="sigmoid",
        cv=3,
    )

    assert isinstance(calibrated_model, CalibratedClassifierCV)
    assert hasattr(calibrated_model, "predict")
    assert hasattr(calibrated_model, "predict_proba")


def test_train_model_fails_fast_on_empty_X(dummy_recipe_and_data):
    _, y_train, preprocessor = dummy_recipe_and_data
    X_empty = pd.DataFrame()

    with pytest.raises(ValueError, match="X_train is empty"):
        train_model(X_empty, y_train, preprocessor, "classification")


def test_train_model_fails_fast_on_row_mismatch(dummy_recipe_and_data):
    X_train, _, preprocessor = dummy_recipe_and_data
    y_mismatch = pd.Series([0, 1])

    with pytest.raises(ValueError, match="do not match y_train rows"):
        train_model(X_train, y_mismatch, preprocessor, "classification")


def test_calibrate_pipeline_fails_without_predict_proba():
    class BadArtifact:
        def predict(self, X):
            return [0] * len(X)

    X_train = pd.DataFrame({"f1": [1, 2, 3, 4]})
    y_train = pd.Series([0, 1, 0, 1])

    with pytest.raises(TypeError, match="predict_proba"):
        calibrate_pipeline(
            pipeline=BadArtifact(),
            X_train=X_train,
            y_train=y_train,
            method="sigmoid",
            cv=2,
        )
