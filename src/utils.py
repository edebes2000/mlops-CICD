# src/utils.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize repetitive I/O (Input/Output) logic.
- Responsibility (separation of concerns): Only basic CSV read/write and model save/load.
- Pipeline contract: Agnostic plumbing. It does not know business logic or pipeline state.

TODO: Replace print statements with standard library logging in a later session
"""

import logging
from pathlib import Path
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: CSV path
    Outputs:
    - df: Loaded DataFrame
    Why this contract matters for reliable ML delivery:
    - Standardized parsing reduces fragile one-off fixes and improves reproducibility
    """
    logger.info(f"Loading CSV from {filepath}")

    if not isinstance(filepath, Path):
        raise TypeError(
            f"filepath must be a pathlib.Path, got type={type(filepath)}")

    if filepath.exists() and not filepath.is_file():
        raise ValueError(
            f"CSV Parsing Error: {filepath} exists but is not a file")

    try:
        df = pd.read_csv(filepath, sep=",")
    except Exception as e:
        raise ValueError(
            f"CSV Parsing Error: Failed to read {filepath}. "
            "Check delimiter, encoding, or file corruption. "
            f"Original pandas error: {e}"
        )

    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save
    - filepath: Output path
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Deterministic saving (index=False) prevents alignment bugs downstream
    """
    logger.info(f"Saving CSV to {filepath}")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: Trained estimator or pipeline
    - filepath: Output path
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Persisted artifacts enable reproducible inference and auditability
    """
    logger.info(f"Saving model to {filepath}")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Model artifact path
    Outputs:
    - model: Deserialized estimator
    Why this contract matters for reliable ML delivery:
    - Fail fast on missing artifacts prevents cryptic inference crashes
    """
    logger.info(f"Loading model from {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {filepath}. "
            "Run the training pipeline first to generate models/model.joblib"
        )

    return joblib.load(filepath)