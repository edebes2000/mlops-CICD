# tests/test_features.py

"""
Educational Goal:
- Why this test exists in an MLOps system: Validate the feature engineering contract without relying on training code
- Responsibility (separation of concerns): Ensure get_feature_preprocessor builds a valid ColumnTransformer and fails fast on bad config
- Pipeline contract (inputs and outputs): Returns a configured ColumnTransformer or raises early when config is invalid
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features import get_feature_preprocessor, _row_sum_numpy


def _transformer_names(preprocessor: ColumnTransformer) -> list:
    """Helper to extract step names from a ColumnTransformer."""
    return [name for name, _, _ in preprocessor.transformers]


# --------------------------------------------------------
# 1) FAIL FAST: Empty configuration raises
# --------------------------------------------------------
def test_get_feature_preprocessor_raises_on_empty_config():
    """
    Inputs:
    - All feature lists empty
    Outputs:
    - ValueError
    Why this contract matters for reliable ML delivery:
    - Empty recipes should crash at build time, not halfway through training
    """
    with pytest.raises(ValueError, match=r"No feature columns configured"):
        get_feature_preprocessor(
            quantile_bin_cols=[],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=[],
            binary_sum_cols=[],
            n_bins=4,
        )


# --------------------------------------------------------
# 2) FAIL FAST: Invalid bin count raises
# --------------------------------------------------------
def test_get_feature_preprocessor_raises_on_invalid_n_bins():
    """
    Inputs:
    - n_bins = 1
    Outputs:
    - ValueError
    Why this contract matters for reliable ML delivery:
    - We fail fast with a clear message before scikit-learn errors later
    """
    with pytest.raises(ValueError, match=r"n_bins must be >= 2"):
        get_feature_preprocessor(
            quantile_bin_cols=["rx_ds"],
            n_bins=1,
        )


# --------------------------------------------------------
# 3) CONTRACT: Returns ColumnTransformer and registers expected blocks
# --------------------------------------------------------
def test_get_feature_preprocessor_returns_columntransformer_and_registers_blocks():
    """
    Inputs:
    - Valid configuration with multiple transformer types
    Outputs:
    - ColumnTransformer with expected named blocks
    Why this contract matters for reliable ML delivery:
    - A stable recipe boundary lets train.py stay simple and consistent
    """
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["rx_ds"],
        categorical_onehot_cols=["gender"],
        numeric_passthrough_cols=["age"],
        binary_sum_cols=["flag_1", "flag_2"],
        n_bins=4,
    )

    assert isinstance(preprocessor, ColumnTransformer)

    names = _transformer_names(preprocessor)

    assert "quantile_bins" in names
    assert "cat_ohe" in names
    assert "binary_sum" in names
    assert "num_scaled" in names


# --------------------------------------------------------
# 4) CORRECTNESS: binary_sum math works (unit test)
# --------------------------------------------------------
def test_row_sum_numpy_computes_expected_sums():
    """
    Inputs:
    - Dummy indicator matrix
    Outputs:
    - Expected row-wise sums as a 2D array
    Why this contract matters for reliable ML delivery:
    - The derived feature must be deterministic before any scaling happens
    """
    df_dummy = pd.DataFrame(
        {
            "flag_1": [1, 0, 1],
            "flag_2": [1, 0, 0],
            "flag_3": [0, 0, 1],
        }
    )

    sums = _row_sum_numpy(df_dummy)

    expected = np.array([[2], [0], [2]])

    assert sums.shape == expected.shape
    np.testing.assert_array_equal(sums, expected)
