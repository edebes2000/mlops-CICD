# test_clean_data.py
"""
Educational Goal:
- Verify that the stateless data cleaning rules work as intended
- Keep tests isolated from file I/O to test only the cleaning logic
"""

import pandas as pd
from src.clean_data import clean_dataframe

TARGET_COLUMN = "OD"


def test_clean_dataframe_happy_path_contract():
    df_messy = pd.DataFrame(
        {
            "ID": [1, 2, 3, 1],
            "rx ds": [10, 20, pd.NA, 10],
            "OD": [0, 1, 0, 0],
        }
    )

    df_clean = clean_dataframe(df_messy, target_column=TARGET_COLUMN)

    assert "rx_ds" in df_clean.columns
    assert "rx ds" not in df_clean.columns

    assert "ID" in df_clean.columns

    assert TARGET_COLUMN in df_clean.columns
    assert df_clean[TARGET_COLUMN].isna().sum() == 0

    assert df_clean["rx_ds"].isna().sum() == 1

    assert list(df_clean.index) == list(range(len(df_clean)))

    assert len(df_clean) == 3


def test_clean_dataframe_inference_mode_does_not_require_target():
    df_infer = pd.DataFrame(
        {
            "ID": [10, 11],
            "rx ds": [5, 7],
            "A": [0, 1],
        }
    )

    df_clean = clean_dataframe(df_infer, target_column=None)

    assert "rx_ds" in df_clean.columns
    assert "ID" in df_clean.columns
    assert "OD" not in df_clean.columns
    assert len(df_clean) == 2
