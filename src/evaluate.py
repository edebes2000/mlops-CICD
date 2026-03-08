# src/evaluate.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Provide consistent evaluation to compare runs and prevent regressions
- Responsibility (separation of concerns): Only computes metrics, no training or artifact writing
- Pipeline contract: Inputs are a fitted model and evaluation data, output is a dictionary of metrics

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score

logger = logging.getLogger(__name__)

def _normalize_problem_type(problem_type: Optional[str]) -> str:
    """
    Inputs
    - problem_type: Raw problem type string

    Outputs
    - normalized: "classification" or "regression"

    Why this contract matters for reliable ML delivery
    - Strict normalization avoids silent configuration errors and makes failures actionable
    """
    return (problem_type or "").strip().lower()


def evaluate_model(model, X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str) -> Dict[str, float]:
    """
    Inputs
    - model: Fitted model or Pipeline with predict()
    - X_eval: Evaluation features (use validation split during development)
    - y_eval: Evaluation target
    - problem_type: "regression" or "classification"

    Outputs
    - metrics: Dictionary of metrics as Python floats

    Why this contract matters for reliable ML delivery
    - Standardized metric keys enable automated quality gates later in continuous integration pipelines
    - Returning JSON safe floats prevents serialization issues in experiment tracking tools
    """
    logger.info("Starting evaluation")

    if X_eval is None or len(X_eval) == 0:
        raise ValueError("Fatal: X_eval is empty. Cannot evaluate model")

    if y_eval is None or len(y_eval) == 0:
        raise ValueError("Fatal: y_eval is empty. Cannot evaluate model")

    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"Fatal: X_eval rows ({len(X_eval)}) do not match y_eval rows ({len(y_eval)})")

    if not hasattr(model, "predict"):
        raise TypeError(
            f"Fatal: model must implement predict(), got type={type(model)}")

    pt = _normalize_problem_type(problem_type)

    if pt == "classification":
        if y_eval.nunique(dropna=True) < 2:
            raise ValueError(
                "Fatal: y_eval contains only one class in this split, so AUC metrics are undefined. "
                "Use stratified splitting, adjust split ratios, or increase dataset size"
            )

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "Fatal: classification model must implement predict_proba()")

        proba = model.predict_proba(X_eval)

        if not isinstance(proba, np.ndarray):
            proba = np.asarray(proba)

        if proba.ndim != 2 or proba.shape[0] != len(X_eval):
            raise ValueError(
                f"Fatal: predict_proba returned invalid shape {getattr(proba, 'shape', None)}")

        if proba.shape[1] < 2:
            raise ValueError(
                "Fatal: predict_proba returned only one probability column. "
                "This usually means the model saw only one class during training"
            )

        y_prob = proba[:, 1]

        metrics = {
            "pr_auc": float(average_precision_score(y_eval, y_prob)),
            "roc_auc": float(roc_auc_score(y_eval, y_prob)),
        }
        logger.info(f"Metrics={metrics}")
        return metrics

    if pt == "regression":
        y_pred = model.predict(X_eval)
        metrics = {"rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred)))}
        logger.info(f"Metrics={metrics}")
        return metrics

    raise ValueError(
        f"Fatal: Unsupported problem_type '{problem_type}'. Use 'classification' or 'regression'")