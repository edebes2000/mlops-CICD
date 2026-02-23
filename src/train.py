# src/train.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Encapsulate training so models are reproducible and swappable without rewiring the pipeline
- Responsibility (separation of concerns): Combines the feature recipe and algorithm into a Pipeline
- Pipeline contract: Inputs are train split, problem type, and preprocessor. Output is a fully fitted Pipeline artifact

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline


def _normalize_problem_type(problem_type: Optional[str]) -> str:
    """
    Inputs:
    - problem_type: Raw problem type string
    Outputs:
    - normalized: "classification" or "regression"
    Why this contract matters for reliable ML delivery:
    - A strict normalization avoids silent configuration errors and makes failures actionable
    """
    normalized = (problem_type or "").strip().lower()
    return normalized


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
) -> Pipeline:
    """
    Inputs:
    - X_train: Training features
    - y_train: Training target
    - preprocessor: The configured ColumnTransformer recipe (not fitted)
    - problem_type: "regression" or "classification"
    Outputs:
    - pipeline: Trained scikit-learn Pipeline object

    Why this contract matters for reliable ML delivery:
    - The preprocessor is fitted only inside pipeline.fit on X_train, preventing leakage from X_test
    """
    print(
        # TODO: replace with logging later
        f"[train.train_model] Training model pipeline for problem_type={problem_type}")

    pt = _normalize_problem_type(problem_type)

    # Fail fast on obvious mismatch
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"Fatal: preprocessor must be a ColumnTransformer. Got type={type(preprocessor)}"
        )

    if pt == "classification":
        # TODO_STUDENT: set model hyperparameters from SETTINGS/config.yml later
        model = LogisticRegression(max_iter=500)
    elif pt == "regression":
        model = LinearRegression()
    else:
        raise ValueError(
            f"Fatal: Unsupported problem_type '{problem_type}'. Use 'classification' or 'regression'."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline
