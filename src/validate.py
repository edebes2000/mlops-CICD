# src/validate.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Fail fast on obvious data issues before expensive training steps run
- Responsibility (separation of concerns): Only validation checks, no cleaning, training, or metrics logic
- Pipeline contract (inputs and outputs): Input is DataFrame and required columns, output is True or a raised error

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: DataFrame to validate
    - required_columns: Columns that must exist
    Outputs:
    - is_valid: True if valid, otherwise raises ValueError
    Why this contract matters for reliable ML delivery:
    - Early validation prevents silent failures and reduces wasted compute and debugging time
    """
    print("[validate.validate_dataframe] Validating dataframe")  # TODO: replace with logging later

    if df is None or len(df) == 0:
        raise ValueError("Validation failed: DataFrame is empty, cannot proceed")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: Missing required columns: {missing}")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace or extend the baseline logic here
    # Why: Validation rules depend on business constraints and what can break downstream training
    # Examples:
    # 1. Ensure target has no missing values
    # 2. Check value ranges for key features
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return True