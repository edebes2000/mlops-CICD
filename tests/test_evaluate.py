# tests/test_evaluate.py
"""
Educational Goal:
- Why this test exists in an MLOps system: Ensure the evaluation module strictly honors the pipeline contract.
- Responsibility: Verify that valid inputs produce mathematically sound metrics, and invalid inputs crash early.
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.evaluate import evaluate_model


@pytest.fixture
def dummy_classification_data():
    """Provides a minimal dataset for classification testing."""
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1], name="target")

    # DummyClassifier creates a fast, baseline model predicting the most frequent class
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    return model, X, y


@pytest.fixture
def dummy_regression_data():
    """Provides a minimal dataset for regression testing."""
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 1.0, 1.0]})
    y = pd.Series([0.0, 1.0, 2.0, 3.0], name="target")

    # DummyRegressor creates a fast, baseline model predicting the mean
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)

    return model, X, y


# --------------------------------------------------------
# 1) HAPPY PATH: Mathematical Soundness
# --------------------------------------------------------
def test_evaluate_classification_returns_float_in_range(dummy_classification_data):
    """The F1 score must be a float strictly between 0.0 and 1.0."""
    model, X, y = dummy_classification_data

    metric = evaluate_model(model=model, X_eval=X,
                            y_eval=y, problem_type="classification")

    assert isinstance(metric, float)
    assert 0.0 <= metric <= 1.0


def test_evaluate_regression_returns_non_negative_float(dummy_regression_data):
    """The RMSE must be a float strictly greater than or equal to 0.0."""
    model, X, y = dummy_regression_data

    metric = evaluate_model(model=model, X_eval=X,
                            y_eval=y, problem_type="regression")

    assert isinstance(metric, float)
    assert metric >= 0.0


# --------------------------------------------------------
# 2) CONTRACT CHECKS: Duck Typing & Configurations
# --------------------------------------------------------
def test_raises_if_model_has_no_predict(dummy_classification_data):
    """Crash immediately if the loaded artifact cannot make predictions."""
    _, X, y = dummy_classification_data

    class BadArtifact:
        pass  # Missing .predict()

    with pytest.raises(TypeError, match="Fatal: model must implement predict"):
        evaluate_model(model=BadArtifact(), X_eval=X,
                       y_eval=y, problem_type="classification")


def test_raises_on_unsupported_problem_type(dummy_classification_data):
    """Crash if configuration routes to an unknown metric."""
    model, X, y = dummy_classification_data

    with pytest.raises(ValueError, match="Fatal: Unsupported problem_type"):
        evaluate_model(model=model, X_eval=X, y_eval=y,
                       problem_type="clustering")


# --------------------------------------------------------
# 3) FAIL FAST: Empty Data & Shape Guardrails
# --------------------------------------------------------
def test_raises_on_none_or_empty_data(dummy_classification_data):
    """Crash if evaluation receives None or 0 rows of data."""
    model, X, y = dummy_classification_data

    # Test None
    with pytest.raises(ValueError, match="Fatal: X_eval is empty"):
        evaluate_model(model=model, X_eval=None, y_eval=y,
                       problem_type="classification")

    # Test empty DataFrame
    X_empty = X.iloc[0:0]
    with pytest.raises(ValueError, match="Fatal: X_eval is empty"):
        evaluate_model(model=model, X_eval=X_empty, y_eval=y,
                       problem_type="classification")


def test_raises_on_length_mismatch(dummy_classification_data):
    """Crash if the feature rows do not perfectly align with the target labels."""
    model, X, y = dummy_classification_data

    # y is one row shorter than X
    y_short = y.iloc[:-1]

    with pytest.raises(ValueError, match="do not match y_eval rows"):
        evaluate_model(model=model, X_eval=X, y_eval=y_short,
                       problem_type="classification")
