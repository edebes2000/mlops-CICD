# src/load_data.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single, predictable entry point for raw data ingestion
- Responsibility (separation of concerns): Only loads raw data, no cleaning, validation, training, or evaluation
- Pipeline contract (inputs and outputs): Input is a raw CSV path, output is a raw DataFrame

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_csv, save_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path to the raw CSV file
    Outputs:
    - df_raw: DataFrame loaded from raw_data_path, or a generated sample dataset for scaffolding
    Why this contract matters for reliable ML delivery:
    - A stable ingestion interface prevents downstream rewiring when data sources evolve
    """
    print(f"[load_data.load_raw_data] Loading raw data from {raw_data_path}")  # TODO: replace with logging later

    if not raw_data_path.exists():
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)

        print("Sample dataset created for scaffolding only. Replace with your real dataset.")  # TODO: replace with logging later

        rng = np.random.default_rng(seed=42)
        n = 10

        df_sample = pd.DataFrame(
            {
                "feature_1": np.arange(n, dtype=float),
                "feature_2": rng.normal(loc=0.0, scale=1.0, size=n),
            }
        )
        df_sample["target"] = (df_sample["feature_1"] + df_sample["feature_2"] > 4.5).astype(int)

        save_csv(df_sample, raw_data_path)

    df_raw = load_csv(raw_data_path)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace CSV loading with real ingestion logic
    # Why: Real systems may load from databases, feature stores, APIs, or partitioned files
    # Examples:
    # 1. Load multiple files and concatenate
    # 2. Enforce a schema and parse dates
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_raw