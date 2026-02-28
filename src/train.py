# src/train.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Encapsulate training so models are reproducible and swappable without rewiring the pipeline
- Responsibility (separation of concerns): Combines the feature recipe and algorithm into a single Pipeline artifact
- Pipeline contract: Inputs are the train split, problem type, and preprocessor. Output is a fully fitted Pipeline artifact

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
    - Strict normalization avoids silent configuration errors and makes failures actionable
    """
    return (problem_type or "").strip().lower()


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
) -> Pipeline:
    """
    Inputs:
    - X_train: Training features (already split, no target column)
    - y_train: Training target
    - preprocessor: ColumnTransformer recipe (should not be fitted here)
    - problem_type: "classification" or "regression"
    Outputs:
    - pipeline: Trained scikit-learn Pipeline object

    Why this contract matters for reliable ML delivery:
    - Fitting happens on training data only, preventing leakage and inflated performance estimates
    - A single fitted pipeline artifact ensures training and inference run the exact same steps
    """
    print(
        # TODO: replace with logging later
        f"[train.train_model] Training model pipeline for problem_type={problem_type}")

    # 1) Fail-fast structural guardrails
    if X_train is None or len(X_train) == 0:
        raise ValueError("Fatal: X_train is empty. Cannot train a model.")

    if y_train is None or len(y_train) == 0:
        raise ValueError("Fatal: y_train is empty. Cannot train a model.")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Fatal: X_train rows ({len(X_train)}) do not match y_train rows ({len(y_train)})."
        )

    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"Fatal: preprocessor must be a ColumnTransformer. Got type={type(preprocessor)}"
        )

    # 2) Model selection
    pt = _normalize_problem_type(problem_type)

    if pt == "classification":
        # Stable default for small datasets and classroom runs
        model = LogisticRegression(
            max_iter=500,
            solver="liblinear",
            random_state=42,
        )
    elif pt == "regression":
        model = LinearRegression()
    else:
        raise ValueError(
            f"Fatal: Unsupported problem_type '{problem_type}'. Use 'classification' or 'regression'."
        )

    # 3) Build the deployable artifact
    # We bundle preprocessing rules and the model into one object that can be saved and reused in inference
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # 4) Execute training
    # This is the only place where .fit() is called in the system
    pipeline.fit(X_train, y_train)

    return pipeline
