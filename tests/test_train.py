"""
Educational Goal:
- Why this test exists in an MLOps system: Confirm train.py produces one fitted Pipeline artifact
- Responsibility (separation of concerns): Test training without depending on features.py
- Pipeline contract: Valid inputs return a Pipeline, invalid inputs raise clear errors
"""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.train import train_model


@pytest.fixture
def dummy_recipe_and_data():
    # Small, deterministic dataset for fast tests
    X_train = pd.DataFrame({"age": [25, 30, 35], "dose": [10, 20, 30]})
    y_train = pd.Series([0, 1, 0])

    # Dummy recipe that simply passes columns through
    # This keeps the test isolated from features.py logic
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
