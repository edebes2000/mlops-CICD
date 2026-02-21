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
    - df_clean: Cleaned DataFrame (baseline is identity transform)
    Why this contract matters for reliable ML delivery:
    - Keeping transformations consistent reduces training serving skew and improves reproducibility
    """
    print("[clean_data.clean_dataframe] Cleaning dataframe")  # TODO: replace with logging later

    df_clean = df_raw.copy()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace or extend the baseline logic here
    # Why: Cleaning depends on data quality and business meaning, so it varies by dataset
    # Examples:
    # 1. Handle missing values and outliers
    # 2. Encode categorical variables and scale numeric features
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    if target_column not in df_clean.columns:
        print(f"[clean_data.clean_dataframe] Warning: target column '{target_column}' not found")  # TODO: replace with logging later

    return df_clean