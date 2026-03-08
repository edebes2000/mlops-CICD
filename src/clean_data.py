# src/clean_data.py
import logging
from typing import Optional
import pandas as pd

"""
Educational Goal:
- Why this module exists in an MLOps system: To isolate dataset-specific formatting and quality filtering from model training, ensuring upstream data anomalies don't cause downstream pipeline crashes.
- Responsibility (separation of concerns): Pure data transformations (standardizing column names, deduplication, dropping missing values) and data observability (logging dropped rows). Strictly NO file I/O and NO model fitting.
- Pipeline contract (inputs and outputs): Inputs are the raw Pandas DataFrame and the target column name. Output is a cleaned Pandas DataFrame that is guaranteed to have the target column, ready for validation and splitting.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

logger = logging.getLogger(__name__)

def clean_dataframe(df_raw: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    One cleaner for both training and inference

    Training mode (target_column provided)
    - Standardize headers
    - Drop exact duplicates
    - Drop rows with missing target

    Inference mode (target_column None)
    - Standardize headers
    - Drop exact duplicates
    - Do not require or drop based on target
    """
    logger.info("Cleaning dataframe")

    if df_raw is None:
        raise ValueError(
            "df_raw is None. Check src/load_data.py and RAW_DATA_PATH in src/main.py")

    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(
            f"df_raw must be a pandas DataFrame, got type={type(df_raw)}")

    df_clean = df_raw.copy()
    initial_rows = len(df_clean)

    # Standardize headers to keep downstream contracts stable
    df_clean.columns = (
        df_clean.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    # Teaching note
    # This drops exact duplicates across all columns (including ID if present)
    # If ID is always unique, duplicates will not be removed, which is expected
    df_clean = df_clean.drop_duplicates()

    if target_column is not None:
        # Standardize target name to match standardized headers
        target_column_std = (
            (target_column or "")
            .strip()
            .replace(" ", "_")
        )

        if not target_column_std:
            raise ValueError("target_column is empty after standardization")

        # Be forgiving to case drift in student datasets
        cols_lower = {c.lower(): c for c in df_clean.columns}
        if target_column_std not in df_clean.columns:
            if target_column_std.lower() in cols_lower:
                target_column_std = cols_lower[target_column_std.lower()]
            else:
                raise ValueError(
                    f"Fatal: target column '{target_column}' missing after cleaning. "
                    "Check SETTINGS['target_column'] in src/main.py and your raw CSV headers"
                )

        # Supervised learning requires a target label for every training row
        df_clean = df_clean.dropna(subset=[target_column_std])

    df_clean = df_clean.reset_index(drop=True)

    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows")

    logger.info(f"Rows after cleaning: {len(df_clean)}")
    return df_clean