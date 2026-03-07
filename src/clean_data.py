from typing import Optional
import pandas as pd


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
    print("[clean_data.clean_dataframe] Cleaning dataframe")  # TODO: replace with logging later

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
        # TODO
        print(f"[clean_data.clean_dataframe] Dropped {dropped_rows} rows")

    # TODO
    print(f"[clean_data.clean_dataframe] Rows after cleaning: {len(df_clean)}")
    return df_clean
