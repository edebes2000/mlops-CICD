# src/validate.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Fail fast on obvious data issues before expensive steps run
- Responsibility (separation of concerns): Only validation checks, no cleaning, features, training, or metrics logic
- Pipeline contract (inputs and outputs): Input is a DataFrame and constraint lists, output is True or a raised ValueError

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Any, List, Optional
import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    check_missing_values: bool = True,
    target_column: Optional[str] = None,
    target_allowed_values: Optional[List[Any]] = None,
    numeric_non_negative_cols: Optional[List[str]] = None,
) -> bool:
    """
    Inputs:
    - df: DataFrame to validate
    - required_columns: Columns that must exist for downstream steps
    - check_missing_values: If True, fail when missing values exist in required columns
    - target_column: Optional target column name for target checks
    - target_allowed_values: Optional list of allowed target values (classification)
    - numeric_non_negative_cols: Optional list of numeric columns that must be >= 0
    Outputs:
    - is_valid: True if valid, otherwise raises ValueError

    Why this contract matters for reliable ML delivery:
    - Validation catches broken assumptions early, reducing wasted compute and debugging time
    """
    print("[validate.validate_dataframe] Validating dataframe")  # TODO: replace with logging later

    # 1) Structural checks
    if df is None:
        raise ValueError(
            "Validation failed: df is None. Check src/load_data.py and your RAW_DATA_PATH")

    if df.empty:
        raise ValueError(
            "Validation failed: DataFrame is empty after cleaning")

    if required_columns is None or len(required_columns) == 0:
        raise ValueError(
            "Validation failed: required_columns is empty. Provide the minimal schema for downstream steps")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Validation failed: Missing required columns: {missing}")

    # 2) Missing value checks
    if check_missing_values:
        cols_with_missing = []
        for col in required_columns:
            if df[col].isna().any():
                cols_with_missing.append(col)

        if cols_with_missing:
            raise ValueError(
                f"Validation failed: Missing values found in required columns: {cols_with_missing}. "
                "Fix in clean_data.py or disable check_missing_values"
            )

    # 3) Target checks (generic, parameter-driven)
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(
                f"Validation failed: target_column '{target_column}' not found in DataFrame")

        if check_missing_values and df[target_column].isna().any():
            raise ValueError(
                f"Validation failed: target column '{target_column}' contains missing values")

        if target_allowed_values is not None:
            actual_values = set(df[target_column].dropna().unique().tolist())
            allowed_values = set(target_allowed_values)

            if not actual_values.issubset(allowed_values):
                raise ValueError(
                    f"Validation failed: Target '{target_column}' has invalid values {sorted(actual_values)}. "
                    f"Expected subset of {sorted(allowed_values)}"
                )

    # 4) Numeric constraints (generic, parameter-driven)
    numeric_non_negative_cols = numeric_non_negative_cols or []
    for col in numeric_non_negative_cols:
        if col not in df.columns:
            raise ValueError(
                f"Validation failed: numeric_non_negative_cols includes missing column '{col}'")

        # Defensive Check: Ensure the column is actually numeric before checking for < 0
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Validation failed: Column '{col}' is in numeric_non_negative_cols but is not a numeric type."
            )

        # Negative value check (Properly indented at the same level as the TypeError check)
        if (df[col] < 0).any():
            raise ValueError(
                f"Validation failed: Column '{col}' contains negative values")

    return True
