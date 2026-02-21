# src/utils.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize file input and output so the pipeline is reproducible and easy to debug
- Responsibility (separation of concerns): Only handles reading and writing artifacts, no data cleaning, training, or metrics
- Pipeline contract (inputs and outputs): DataFrames and models go in and out via explicit file paths

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to a CSV file on disk
    Outputs:
    - df: DataFrame loaded from disk
    Why this contract matters for reliable ML delivery:
    - Standardized I/O reduces notebook drift and makes failures easier to reproduce
    """
    print(f"[utils.load_csv] Loading CSV from {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust read_csv parameters for your dataset
    # Why: Real CSVs often need delimiter, encoding, or dtype hints
    # Examples:
    # 1. pd.read_csv(filepath, sep=";")
    # 2. pd.read_csv(filepath, encoding="utf-8")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save
    - filepath: Path where the CSV should be written
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Persisting intermediate artifacts enables reproducibility, debugging, and auditing
    """
    print(f"[utils.save_csv] Saving CSV to {filepath}")  # TODO: replace with logging later

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust CSV output formatting if required
    # Why: Downstream consumers may require index, specific delimiter, or float formatting
    # Examples:
    # 1. df.to_csv(filepath, index=True)
    # 2. df.to_csv(filepath, index=False, sep=";")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: Fitted model object to serialize
    - filepath: Path where the model should be written
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - A saved model artifact enables repeatable inference and consistent promotion across environments
    """
    print(f"[utils.save_model] Saving model to {filepath}")  # TODO: replace with logging later

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Tune joblib.dump options if needed
    # Why: Compression and protocol choices affect storage cost and load speed
    # Examples:
    # 1. joblib.dump(model, filepath, compress=3)
    # 2. joblib.dump(model, filepath, protocol=4)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to a serialized model artifact
    Outputs:
    - model: Deserialized model object
    Why this contract matters for reliable ML delivery:
    - Loading from a known artifact path standardizes inference and reduces environment specific behavior
    """
    print(f"[utils.load_model] Loading model from {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add lightweight conventions for model versioning
    # Why: Production systems often require predictable naming and lifecycle controls
    # Examples:
    # 1. Check filepath.exists() and raise a clear error
    # 2. Load a versioned model filename
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return joblib.load(filepath)