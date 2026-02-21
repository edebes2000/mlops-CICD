# src/evaluate.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Provide consistent evaluation to compare runs and prevent regressions
- Responsibility (separation of concerns): Only computes metrics, no training or artifact writing
- Pipeline contract (inputs and outputs): Inputs are model and test data, output is a single float metric

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: Fitted model with predict method
    - X_test: Test features
    - y_test: Test target
    - problem_type: "regression" or "classification"
    Outputs:
    - metric_value: RMSE for regression or F1 score for classification
    Why this contract matters for reliable ML delivery:
    - Consistent evaluation supports objective go or no go decisions and reduces quality regressions
    """
    print(f"[evaluate.evaluate_model] Evaluating model for problem_type={problem_type}")  # TODO: replace with logging later

    y_pred = model.predict(X_test)
    pt = (problem_type or "").strip().lower()

    if pt == "classification":
        metric_value = float(f1_score(y_test, y_pred, average="binary"))
        metric_name = "F1"
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metric_value = rmse
        metric_name = "RMSE"

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace or extend the baseline logic here
    # Why: Metrics should reflect business success criteria and error costs
    # Examples:
    # 1. Add Mean Absolute Error for regression
    # 2. Add Precision and Recall for classification
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print(f"[evaluate.evaluate_model] {metric_name}={metric_value}")  # TODO: replace with logging later
    return metric_value