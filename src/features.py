# src/features.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Isolate feature engineering rules from their execution to prevent leakage and drift.
- Responsibility (separation of concerns): Define the preprocessing recipe as a ColumnTransformer. No file I/O, no .fit() calls here.
- Pipeline contract: Inputs are configuration lists. Output is a scikit-learn ColumnTransformer.

TEACHING NOTE - Where do transformations belong?
- Stateful transformations (MUST be fitted on X_train only) belong here. Example: Quantile bin edges.
- Stateless transformations (Math operations) CAN belong here if they are part of the model's unique contract. Example: binary_sum. Putting it here ensures the deployed model calculates it automatically.
- Only put stateless transforms in `clean_data.py` if they are part of a canonical, company-wide data schema used beyond just this model.

Why this prevents leakage:
- The recipe is fitted ONLY inside `pipeline.fit(X_train, y_train)` inside train.py.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import List, Optional
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, FunctionTransformer


def _row_sum_numpy(X: np.ndarray) -> np.ndarray:
    """
    Inputs:
    - X: 2D numpy array of shape (n_rows, n_cols) containing binary or numeric indicators.
    Outputs:
    - sums: 2D numpy array of shape (n_rows, 1) containing row-wise sums.
    """
    sums = np.sum(X, axis=1).reshape(-1, 1)
    return sums


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    binary_sum_cols: Optional[List[str]] = None,
    n_bins: int = 3,
) -> ColumnTransformer:
    """
    Build a preprocessing recipe safely.
    """
    print("[features.get_feature_preprocessor] Building feature recipe from configuration")

    if n_bins < 2:
        raise ValueError("Fatal: n_bins must be >= 2 for quantile binning.")

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []
    binary_sum_cols = binary_sum_cols or []

    if not (quantile_bin_cols or categorical_onehot_cols or numeric_passthrough_cols or binary_sum_cols):
        raise ValueError(
            "Fatal: No feature columns configured for the preprocessor.")

    transformers = []

    if quantile_bin_cols:
        quantile_binner = KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="quantile",
        )
        transformers.append(
            ("quantile_bins", quantile_binner, quantile_bin_cols))

    if categorical_onehot_cols:
        try:
            onehot = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat_ohe", onehot, categorical_onehot_cols))

    if binary_sum_cols:
        # TEACHING NOTE: FunctionTransformer embeds custom logic (like summing) directly into the pipeline artifact.
        binary_sum_transformer = FunctionTransformer(
            func=_row_sum_numpy,
            validate=False,
            feature_names_out=lambda self, input_features: np.array([
                                                                    "binary_sum"]),
        )
        transformers.append(
            ("binary_sum", binary_sum_transformer, binary_sum_cols))

    if numeric_passthrough_cols:
        transformers.append(
            ("num_pass", "passthrough", numeric_passthrough_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor
