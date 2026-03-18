"""
Educational goal
- Encapsulate model training so orchestration (main) stays clean
- Make model choice and hyperparameters configurable via config.yaml
- Return a single fitted Pipeline artifact that can be saved and reused

Design choices for simplicity
- Prefer functions over classes
- Use a small model switch based on model_type
- Keep scikit-learn Pipeline as the core artifact
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _normalize_problem_type(problem_type: Optional[str]) -> str:
    return (problem_type or "").strip().lower()


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Train a scikit-learn Pipeline.

    Inputs
    - X_train: Features for training (target removed)
    - y_train: Target values
    - preprocessor: ColumnTransformer describing the feature recipe
    - problem_type: "classification" or "regression"
    - model_params: dictionary from config.yaml training.<problem_type>

    Output
    - A fitted scikit-learn Pipeline artifact
    """
    logger.info("Training model pipeline for problem_type=%s", problem_type)

    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train is empty. Cannot train a model")
    if y_train is None or len(y_train) == 0:
        raise ValueError("y_train is empty. Cannot train a model")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train rows ({len(X_train)}) do not match y_train rows ({len(y_train)})"
        )
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"preprocessor must be a ColumnTransformer. Got type={type(preprocessor)}"
        )

    pt = _normalize_problem_type(problem_type)

    # Copy so we never mutate the original config dictionary
    params: Dict[str, Any] = dict(model_params) if isinstance(
        model_params, dict) else {}

    # model_type is a routing key, not a constructor argument
    model_type = str(params.pop("model_type", "") or "").strip().lower()

    # Calibration keys belong to orchestration, not the base estimator constructor
    params.pop("calibration_enabled", None)
    params.pop("calibration_method", None)
    params.pop("calibration_cv", None)

    if pt == "classification":
        if not model_type:
            model_type = "logistic_regression"

        if model_type != "logistic_regression":
            raise ValueError(
                f"Unsupported classification model_type '{model_type}'. Supported: 'logistic_regression'"
            )

        try:
            model = LogisticRegression(**params)
        except TypeError as e:
            raise ValueError(
                "Invalid classification hyperparameters in config.yaml under training.classification. "
                f"Constructor error: {e}. "
                "Fix the keys or remove the unsupported parameter."
            ) from e

    elif pt == "regression":
        if not model_type:
            model_type = "linear_regression"

        if model_type != "linear_regression":
            raise ValueError(
                f"Unsupported regression model_type '{model_type}'. Supported: 'linear_regression'"
            )

        try:
            model = LinearRegression(**params)
        except TypeError as e:
            raise ValueError(
                "Invalid regression hyperparameters in config.yaml under training.regression. "
                f"Constructor error: {e}. "
                "Fix the keys or remove the unsupported parameter."
            ) from e

    else:
        raise ValueError(
            "Unsupported problem_type. Use 'classification' or 'regression'")

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    logger.info("Base model training completed")
    return pipeline


def calibrate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
    cv: int = 3,
):
    """
    Calibrate a fitted classification pipeline so predicted probabilities are more reliable.

    Inputs
    - pipeline: A fitted scikit-learn Pipeline with predict_proba support
    - X_train: Training features used for calibration fitting
    - y_train: Training target used for calibration fitting
    - method: Calibration method supported by scikit-learn ("sigmoid" or "isotonic")
    - cv: Number of folds used by CalibratedClassifierCV

    Output
    - A fitted CalibratedClassifierCV object that behaves like a prediction model

    Why this matters in production
    - A classifier can rank cases reasonably well but still output poorly calibrated probabilities
    - Calibration is useful when probabilities drive business thresholds, triage, or intervention rules
    """
    logger.info(
        "Calibrating model probabilities | method=%s | cv=%s", method, cv)

    if pipeline is None:
        raise ValueError("pipeline cannot be None for calibration")
    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train is empty. Cannot calibrate model")
    if y_train is None or len(y_train) == 0:
        raise ValueError("y_train is empty. Cannot calibrate model")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train rows ({len(X_train)}) do not match y_train rows ({len(y_train)})"
        )
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "pipeline must implement predict_proba() for calibration")
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Calibration method must be 'sigmoid' or 'isotonic'")
    if int(cv) < 2:
        raise ValueError("Calibration cv must be at least 2")

    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=int(cv),
    )
    calibrated_model.fit(X_train, y_train)

    logger.info("Model calibration completed")
    return calibrated_model
