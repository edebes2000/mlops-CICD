# src/clean_data.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Isolate dataset specific transformations so they are repeatable and testable
- Responsibility (separation of concerns): Only cleaning and transformation, no file I/O and no model work
- Pipeline contract (inputs and outputs): Input is raw DataFrame, output is cleaned DataFrame ready for splitting

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: Raw DataFrame
    - target_column: Name of the target column
    Outputs:
    - df_clean: Cleaned DataFrame
    Why this contract matters for reliable ML delivery:
    - Keeping transformations consistent reduces training serving skew and improves reproducibility
    - Dropping invalid rows here prevents confusing model failures downstream
    """
    # TODO: replace with logging later
    print("[clean_data.clean_dataframe] Cleaning dataframe")

    if df_raw is None:
        raise ValueError(
            "df_raw is None. Check src/load_data.py and RAW_DATA_PATH in src/main.py")

    df_clean = df_raw.copy()
    initial_rows = len(df_clean)

    # 1) Standardize column names
    # Strip hidden whitespace and remove spaces explicitly
    # e.g. makes "rx ds" become "rx_ds" for stable downstream contracts
    df_clean.columns = df_clean.columns.str.strip().str.replace(" ", "_", regex=False)

    # 2) Drop non predictive identifier columns (idempotent)
    df_clean = df_clean.drop(columns=["ID"], errors="ignore")

    # 3) Remove duplicates and missing values
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna()

    # 4) Reset index to prevent downstream alignment bugs
    df_clean = df_clean.reset_index(drop=True)

    # MLOps Observability: Log data loss and final state
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        # TODO: replace with logging later
        print(
            f"[clean_data.clean_dataframe] Dropped {dropped_rows} rows due to NA or duplicates")
    # TODO: replace with logging later
    print(f"[clean_data.clean_dataframe] Rows after cleaning: {len(df_clean)}")

    # Fail fast: pipeline cannot train without a target column
    if target_column not in df_clean.columns:
        raise ValueError(
            f"Fatal: target column '{target_column}' missing after cleaning. "
            "Check TARGET_COLUMN in src/main.py and your raw CSV headers"
        )

    return df_clean
