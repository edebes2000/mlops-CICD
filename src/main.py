# src/main.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Orchestrate the pipeline in a readable entry point.
- Responsibility (separation of concerns): Coordinates steps, handles the split, injects config, and delegates work to modules
- Pipeline contract (inputs and outputs): Produces clean data and a saved pipeline artifact.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.load_data import load_raw_data
from src.features import get_feature_preprocessor
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

# --------------------------------------------------------
# PATHS & CONFIGURATION
# --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pointing directly to your illustrative demo data
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "opiod_raw_data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# INSTRUCTOR MASTER BRANCH CONFIGURATION:
# This is set up to run your Opioid demo perfectly out of the box.
SETTINGS = {
    "is_example_config": False,
    "target_column": "OD",
    "problem_type": "classification",
    "split": {"test_size": 0.25, "random_state": 42},
    "features": {
        "quantile_bin": ["rx_ds"],
        "categorical_onehot": [],
        "numeric_passthrough": [],
        "binary_sum_cols": [],
        "n_bins": 4,
    },
}


def main():
    print("[main.main] Starting pipeline")

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SETTINGS.get("is_example_config", False):
        raise ValueError(
            "Fatal: SETTINGS is an example. Update target_column and feature lists for YOUR dataset, then set 'is_example_config': False."
        )

    print("[main.main] 1) LOAD")
    df_raw = load_raw_data(RAW_DATA_PATH)

    print("[main.main] 2) CLEAN")
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])

    print("[main.main] 3) SAVE PROCESSED CSV")
    save_csv(df_clean, CLEAN_DATA_PATH)

    # Note: Assuming src/validate.py is fully implemented from your earlier sessions.
    print("[main.main] 4) VALIDATE")
    validate_dataframe(df_clean, required_columns=[SETTINGS["target_column"]])

    print("[main.main] 5) SPLIT")
    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]

    try:
        # Renamed to _X_test and _y_test to signal they are intentionally parked
        X_train, _X_test, y_train, _y_test = train_test_split(
            X, y,
            test_size=SETTINGS["split"]["test_size"],
            random_state=SETTINGS["split"]["random_state"],
            stratify=y if SETTINGS["problem_type"] == "classification" else None
        )
    except ValueError as e:
        print(
            f"[main] Warning: Stratified split failed: {e}. Falling back to random split.")
        X_train, _X_test, y_train, _y_test = train_test_split(
            X, y,
            test_size=SETTINGS["split"]["test_size"],
            random_state=SETTINGS["split"]["random_state"]
        )

    # --- 6) FAIL-FAST FEATURE CHECKS ---
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
                f"Fatal: Column '{col}' must be numeric for quantile binning. Found dtype={X_train[col].dtype}")

    # --- 7) BUILD RECIPE ---
    print("[main.main] 7) BUILD FEATURE RECIPE")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        binary_sum_cols=SETTINGS["features"]["binary_sum_cols"],
        n_bins=SETTINGS["features"]["n_bins"],
    )

    # --- 8) TRAIN PIPELINE ---
    print("[main.main] 8) TRAIN")
    model_pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=SETTINGS["problem_type"]
    )

    # --- 9) SAVE MODEL ---
    print("[main.main] 9) SAVE MODEL")
    save_model(model_pipeline, MODEL_PATH)

    print("[main.main] Done")
    print(f"[main.main] Wrote {CLEAN_DATA_PATH}")
    print(f"[main.main] Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
