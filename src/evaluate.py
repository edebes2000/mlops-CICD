# src/evaluate.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Provide consistent evaluation to compare runs and prevent regressions
- Responsibility (separation of concerns): Only computes metrics, no training or artifact writing
- Pipeline contract: Inputs are a fitted model and evaluation data, output is a single float metric

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error


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


def evaluate_model(model, X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: Fitted model or Pipeline with predict()
    - X_eval: Evaluation features (use Validation split for development)
    - y_eval: Evaluation target
    - problem_type: "regression" or "classification"
    Outputs:
    - metric_value: RMSE for regression or F1 score for classification

    Why this contract matters for reliable ML delivery:
    - Consistent evaluation supports objective go/no-go decisions and reduces quality regressions
    """
    print("[evaluate.evaluate_model] Starting evaluation")  # TODO: replace with logging later

    # 1) Fail-fast structural guardrails
    if X_eval is None or len(X_eval) == 0:
        raise ValueError("Fatal: X_eval is empty. Cannot evaluate model.")
    if y_eval is None or len(y_eval) == 0:
        raise ValueError("Fatal: y_eval is empty. Cannot evaluate model.")
    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"Fatal: X_eval rows ({len(X_eval)}) do not match y_eval rows ({len(y_eval)}).")

    # Enforce Pipeline Contract: The artifact must be able to predict
    if not hasattr(model, "predict"):
        raise TypeError(
            f"Fatal: model must implement predict(), got type={type(model)}")

    # 2) Execute inference
    pt = _normalize_problem_type(problem_type)
    y_pred = model.predict(X_eval)

    # 3) Calculate metric
    if pt == "classification":
        # Assumes target is binary {0,1} as validated in main.py
        metric_value = float(f1_score(y_eval, y_pred, average="binary"))
        # TODO: replace with logging later
        print(f"[evaluate.evaluate_model] Metric=F1 value={metric_value:.4f}")
        return metric_value

    elif pt == "regression":
        # np.sqrt is universally compatible across all scikit-learn versions
        metric_value = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
        # TODO: replace with logging later
        print(
            f"[evaluate.evaluate_model] Metric=RMSE value={metric_value:.4f}")
        return metric_value

    else:
        raise ValueError(
            f"Fatal: Unsupported problem_type '{problem_type}'. Use 'classification' or 'regression'.")
