# src/train.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Encapsulate training so models are reproducible and swappable without rewiring the pipeline
- Responsibility (separation of concerns): Only trains a model, no I/O, cleaning, or evaluation
- Pipeline contract (inputs and outputs): Inputs are train split and problem type, output is a fitted model

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


def train_model(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str):
    """
    Inputs:
    - X_train: Training features
    - y_train: Training target
    - problem_type: "regression" or "classification"
    Outputs:
    - model: Trained model object
    Why this contract matters for reliable ML delivery:
    - A stable training interface supports testing, automation, and later deployment
    """
    print(f"[train.train_model] Training model for problem_type={problem_type}")  # TODO: replace with logging later

    pt = (problem_type or "").strip().lower()
    if pt == "classification":
        model = LogisticRegression(max_iter=200)
    else:
        model = LinearRegression()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace or extend the baseline logic here
    # Why: Model choice and hyperparameters depend on business constraints and data characteristics
    # Examples:
    # 1. Add regularization or change solver for LogisticRegression
    # 2. Swap to tree based models for non linear relationships
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    model.fit(X_train, y_train)
    return model