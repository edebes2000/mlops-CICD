# src/infer.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Separate inference so it can be reused for batch or real-time prediction.
- Responsibility (separation of concerns): Only runs model.predict and formats outputs. No training or metrics.
- Pipeline contract: Inputs are a fitted model and a feature DataFrame. Output is a strictly formatted predictions DataFrame.

TODO: Replace print statements with standard library logging in a later session
TODO: Move any hardcoded configurations to config.yml in a later session
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def run_inference(model, X_infer: pd.DataFrame, *, include_proba: bool = False) -> pd.DataFrame:
    """
    Executes model predictions on new, unseen data.

    Inputs:
    - model: Fitted scikit-learn compatible artifact with a .predict() method
    - X_infer: Pandas DataFrame of inference features
    - include_proba: If True, attempts to return the probability of the positive class.

    Outputs:
    - df_pred: DataFrame containing 'prediction' and optionally 'proba', retaining the original index.

    Why this contract matters for reliable ML delivery:
    - A consistent schema reduces downstream integration risk for APIs or batch jobs.
    - Preserving the index allows predictions to be joined back to the original database records.
    - Defensive shape checking on probabilities prevents silent downstream crashes.
    """
    logger.info("Running inference")

    # 1) Fail-fast structural guardrails
    if X_infer is None or len(X_infer) == 0:
        raise ValueError("Fatal: X_infer is empty. Cannot run inference.")

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError(
            f"Fatal: X_infer must be a pandas DataFrame to retain feature names. Got type={type(X_infer)}")

    # Enforce Pipeline Contract: The artifact must be able to predict
    if not hasattr(model, "predict"):
        raise TypeError(
            f"Fatal: model must implement predict(), got type={type(model)}")

    # 2) Initialize Output DataFrame (Preserving the index for downstream joins)
    df_pred = pd.DataFrame(index=X_infer.index)

    # 3) Execute Inference
    df_pred["prediction"] = model.predict(X_infer)

    # 4) Optionally attach probabilities (Critical for risk-scoring models like OD)
    if include_proba:
        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "Fatal: include_proba=True but the loaded model does not support predict_proba().")

        proba = model.predict_proba(X_infer)

        # Defensive MLOps: Ensure the model returned a valid 2D array matching our row count
        if not isinstance(proba, np.ndarray) or proba.ndim != 2 or proba.shape[0] != len(X_infer):
            raise ValueError(
                f"Fatal: predict_proba returned an invalid shape. Got {getattr(proba, 'shape', None)}")

        # Ensure we actually have a second column (index 1) representing the positive class (OD=1)
        if proba.shape[1] < 2:
            raise ValueError(
                f"Fatal: predict_proba must return at least 2 columns for binary classification. Got {proba.shape}")

        df_pred["proba"] = proba[:, 1].astype(float)

    return df_pred