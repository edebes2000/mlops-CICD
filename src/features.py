# src/features.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Define feature engineering rules as a reusable recipe that can be fitted safely on training data only
- Responsibility (separation of concerns): Build and return a ColumnTransformer recipe. No file I/O, no .fit() calls here
- Pipeline contract (inputs and outputs): Inputs are configuration lists. Output is a scikit-learn ColumnTransformer

Why this prevents leakage:
- The recipe is fitted only inside pipeline.fit(X_train, y_train) in train.py

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import List, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)


def _row_sum_numpy(X) -> np.ndarray:
    """
    Inputs:
    - X: 2D array-like (DataFrame or ndarray) of indicator columns
    Outputs:
    - sums: 2D numpy array of shape (n_rows, 1)

    Why this contract matters for reliable ML delivery:
    - Derived features must be computed the same way in training and inference
    """
    # FunctionTransformer may pass a pandas object, so we convert explicitly
    X_np = np.asarray(X)
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)

    sums = np.sum(X_np, axis=1).reshape(-1, 1)
    return sums


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    binary_sum_cols: Optional[List[str]] = None,
    n_bins: int = 3,
) -> ColumnTransformer:
    """
    Inputs:
    - quantile_bin_cols: Numeric columns to bucket into quantiles
    - categorical_onehot_cols: Categorical columns to one hot encode
    - numeric_passthrough_cols: Numeric columns to keep as numeric features
    - binary_sum_cols: Indicator columns to sum into one derived feature
    - n_bins: Number of quantile buckets
    Outputs:
    - preprocessor: Unfitted ColumnTransformer recipe

    Why this contract matters for reliable ML delivery:
    - This function defines the recipe only. Fitting happens later on X_train only
    """
    print("[features.get_feature_preprocessor] Building feature recipe from configuration")  # TODO: replace with logging later

    if n_bins < 2:
        raise ValueError("Fatal: n_bins must be >= 2 for quantile binning")

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []
    binary_sum_cols = binary_sum_cols or []

    if not (quantile_bin_cols or categorical_onehot_cols or numeric_passthrough_cols or binary_sum_cols):
        raise ValueError(
            "Fatal: No feature columns configured for the preprocessor")

    transformers = []

    # 1) Quantile features: Impute -> Quantile bin -> Scale
    # We use encode="ordinal" so bins become numeric codes (0, 1, 2, ...)
    # Then scaling makes these codes comparable to other numeric features
    if quantile_bin_cols:
        quantile_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("qbin", KBinsDiscretizer(n_bins=n_bins,
                 encode="ordinal", strategy="quantile")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(
            ("quantile_bins", quantile_pipe, quantile_bin_cols))

    # 2) Categorical features: Impute -> One hot encode
    # handle_unknown="ignore" prevents crashes when new categories appear at inference time
    if categorical_onehot_cols:
        try:
            onehot = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # Older scikit-learn versions
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", onehot),
            ]
        )
        transformers.append(("cat_ohe", cat_pipe, categorical_onehot_cols))

    # 3) Derived feature: binary_sum, treated as a numeric feature
    # We compute it inside the pipeline so training and inference always match
    # feature_names_out is used only to give the derived column a stable name in debugging and tests
    if binary_sum_cols:
        binary_sum_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("sum", FunctionTransformer(
                    func=_row_sum_numpy,
                    validate=False,
                    feature_names_out=lambda self, input_features: np.array([
                                                                            "binary_sum"]),
                )),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("binary_sum", binary_sum_pipe, binary_sum_cols))

    # 4) Standard numeric features: Impute -> Scale
    if numeric_passthrough_cols:
        num_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num_scaled", num_pipe, numeric_passthrough_cols))

    # Gatekeeper: only configured columns enter the model
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor
