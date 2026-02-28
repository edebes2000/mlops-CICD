# src/main.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Orchestrate the pipeline in a readable entry point
- Responsibility (separation of concerns): Coordinate steps, handle splits, inject configuration, and delegate work to modules
- Pipeline contract (inputs and outputs): Produces a cleaned dataset and a trained pipeline artifact saved to disk

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.features import get_feature_preprocessor
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe
from src.evaluate import evaluate_model

# --------------------------------------------------------
# PATHS & CONFIGURATION
# --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "opiod_raw_data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# This list is defined once to avoid copy paste mistakes
# In a later session, this moves to config.yml
BINARY_SUM_COLS = [
    "A", "B", "C", "D", "E", "F",
    "H", "I", "J", "K", "L", "M", "N",
    "R", "S", "T",
    "Low_inc", "SURG",
]

# Instructor demo configuration
# Students will replace these lists for their own dataset
SETTINGS = {
    "is_example_config": False,
    "target_column": "OD",
    "problem_type": "classification",
    # 3 way split: train, validation, test
    # With 1000 rows, this yields about 800, 150, 50 rows
    "split": {"test_size": 0.05, "val_size": 0.15, "random_state": 42},
    "features": {
        "quantile_bin": ["rx_ds"],
        "categorical_onehot": [],
        "numeric_passthrough": [],
        "binary_sum_cols": BINARY_SUM_COLS,
        "n_bins": 4,
    },
    "validation": {
        # Keep this separate from features to avoid accidental assumptions
        "numeric_non_negative_cols": ["rx_ds"],
    },
}


def _three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
):
    """
    Inputs:
    - X: Feature table
    - y: Target vector
    - test_size: Fraction reserved for the final test vault
    - val_size: Fraction reserved for validation during development
    - random_state: Reproducible split seed
    - stratify: If True, preserve class ratios in each split
    Outputs:
    - X_train, X_val, X_test, y_train, y_val, y_test

    Why this contract matters for reliable ML delivery:
    - Train learns, validation guides decisions, test audits the final result
    """
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
        raise ValueError(
            "Fatal: split sizes must satisfy 0 < test_size, 0 < val_size, and test_size + val_size < 1")

    stratify_y = y if stratify else None

    try:
        # Step A: carve out the test set first
        # This keeps a small untouched vault for the final audit
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )

        # Step B: split the remaining data into train and validation
        # We want val_size of the total dataset, taken from the remaining (1 - test_size)
        relative_val_size = val_size / (1.0 - test_size)
        stratify_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=stratify_temp,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except ValueError as e:
        # TODO: replace with logging later
        print(
            f"[main] Warning: Stratified split failed: {e}. Falling back to random split.")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    print("[main.main] Starting pipeline")  # TODO: replace with logging later

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SETTINGS.get("is_example_config", False):
        raise ValueError(
            "Fatal: SETTINGS is an example. Update target_column and feature lists for YOUR dataset, then set 'is_example_config': False."
        )

    # 1) LOAD
    print("[main.main] 1) LOAD")  # TODO: replace with logging later
    df_raw = load_raw_data(RAW_DATA_PATH)

    # 2) CLEAN
    print("[main.main] 2) CLEAN")  # TODO: replace with logging later
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])

    # 3) SAVE PROCESSED CSV
    # Saving the cleaned dataset is useful for debugging and reproducibility
    # TODO: replace with logging later
    print("[main.main] 3) SAVE PROCESSED CSV")
    save_csv(df_clean, CLEAN_DATA_PATH)

    # 4) VALIDATE
    # Validation is a security gate to catch missing columns or empty data early
    print("[main.main] 4) VALIDATE")  # TODO: replace with logging later

    required_columns = (
        [SETTINGS["target_column"]]
        + SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + SETTINGS["features"]["binary_sum_cols"]
    )

    validate_dataframe(
        df=df_clean,
        required_columns=required_columns,
        check_missing_values=True,
        target_column=SETTINGS["target_column"],
        target_allowed_values=[
            0, 1] if SETTINGS["problem_type"] == "classification" else None,
        numeric_non_negative_cols=SETTINGS["validation"]["numeric_non_negative_cols"],
    )

    # 5) SPLIT INTO TRAIN, VALIDATION, TEST
    # TODO: replace with logging later
    print("[main.main] 5) SPLIT INTO TRAIN, VALIDATION, TEST")

    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]

    X_train, X_val, X_test, y_train, y_val, y_test = _three_way_split(
        X,
        y,
        test_size=SETTINGS["split"]["test_size"],
        val_size=SETTINGS["split"]["val_size"],
        random_state=SETTINGS["split"]["random_state"],
        stratify=(SETTINGS["problem_type"] == "classification"),
    )

    print("[main.main] Split sizes")  # TODO: replace with logging later
    print("Train:", X_train.shape, "Validation:",
          X_val.shape, "Test:", X_test.shape)

    # 6) FAIL FAST FEATURE CHECKS
    # These checks catch common classroom misconfigurations early
    configured_cols = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + SETTINGS["features"]["binary_sum_cols"]
    )
    if not configured_cols:
        raise ValueError(
            "Fatal: No feature columns configured in SETTINGS['features'].")

    missing = set(configured_cols) - set(X_train.columns)
    if missing:
        raise ValueError(
            f"Fatal: Configured columns not found in dataset: {sorted(missing)}")

    for col in SETTINGS["features"]["quantile_bin"]:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            raise ValueError(
                f"Fatal: Column '{col}' must be numeric for quantile binning. Found dtype={X_train[col].dtype}"
            )

    # 7) BUILD FEATURE RECIPE
    # We build the blueprint here
    # Fitting happens only inside train_model on X_train
    # TODO: replace with logging later
    print("[main.main] 7) BUILD FEATURE RECIPE")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        binary_sum_cols=SETTINGS["features"]["binary_sum_cols"],
        n_bins=SETTINGS["features"]["n_bins"],
    )

    # 8) TRAIN
    # train_model is the only place where .fit() happens
    print("[main.main] 8) TRAIN")  # TODO: replace with logging later
    model_pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=SETTINGS["problem_type"],
    )

    # 8.5) EVALUATE (Using the Validation split)
    # Validation guides decisions during development; the Test vault remains mostly untouched.
    print("[main.main] 8.5) EVALUATE")  # TODO: replace with logging later
    val_metric = evaluate_model(
        model=model_pipeline,
        X_eval=X_val,
        y_eval=y_val,
        problem_type=SETTINGS["problem_type"],
    )
    print(f"[main.main] Validation metric={val_metric:.4f}")  # TODO: replace with logging later

    # 9) SAVE MODEL
    # We save the full pipeline artifact, not just the estimator
    # This prevents training serving skew because preprocessing is bundled inside
    print("[main.main] 9) SAVE MODEL")  # TODO: replace with logging later
    save_model(model_pipeline, MODEL_PATH)

    # 10) QUICK INFERENCE DEMO USING TEST VAULT
    # We keep test mostly untouched, but sampling 10 rows is a safe teaching demo
    # TODO: replace with logging later
    print("[main.main] 10) INFERENCE DEMO (10 ROWS FROM TEST)")
    X_infer = X_test.sample(
        n=10, random_state=SETTINGS["split"]["random_state"])
    preds = model_pipeline.predict(X_infer)
    print("[main.main] First 10 predictions:", preds.tolist())

    print("[main.main] Done")  # TODO: replace with logging later
    print(f"[main.main] Wrote {CLEAN_DATA_PATH}")
    print(f"[main.main] Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
