# src/utils.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize repetitive I/O (Input/Output) logic.
- Responsibility (separation of concerns): Only basic CSV read/write and model save/load.
- Pipeline contract: Agnostic plumbing. It does not know business logic or pipeline state.

TODO: Replace print statements with standard library logging in a later session
"""

from pathlib import Path
import pandas as pd
import joblib


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs: filepath (Path)
    Outputs: df (DataFrame)
    Why this matters: Standardized parsing catches encoding/delimiter issues universally.
    """
    print(f"[utils.load_csv] Loading CSV from {filepath}")  # TODO: replace with logging later

    try:
        # Explicit separator ensures determinism
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
    Inputs: df (DataFrame), filepath (Path)
    Outputs: None
    Why this matters: Deterministic saving (index=False) prevents alignment bugs downstream.
    """
    print(f"[utils.save_csv] Saving CSV to {filepath}")  # TODO: replace with logging later
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs: model (sklearn estimator), filepath (Path)
    Outputs: None
    Why this matters: Persisting models enables reproducible inference and deployment.
    """
    print(f"[utils.save_model] Saving model to {filepath}")  # TODO: replace with logging later
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs: filepath (Path)
    Outputs: model (Deserialized estimator)
    Why this matters: Fail-fast on missing artifacts prevents cryptic inference crashes.
    """
    print(f"[utils.load_model] Loading model from {filepath}")  # TODO: replace with logging later

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {filepath}. "
            "Run the training pipeline first to generate models/model.joblib"
        )

    return joblib.load(filepath)