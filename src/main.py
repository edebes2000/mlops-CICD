# src/main.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Orchestrate the pipeline in a readable entry point that runs the same locally and in automation
- Responsibility (separation of concerns): Coordinates steps and artifact paths, delegates work to modules
- Pipeline contract (inputs and outputs): Produces clean data, a saved model, and a saved predictions report

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------

RAW_DATA_PATH = Path("data/raw/sample.csv")
TARGET_COLUMN = "target"
PROBLEM_TYPE = "regression"

CLEAN_DATA_PATH = Path("data/processed/clean.csv")
MODEL_PATH = Path("models/model.joblib")
PREDICTIONS_PATH = Path("reports/predictions.csv")


def main():
    """
    Inputs:
    - None (uses module level configuration for notebook friendliness)
    Outputs:
    - None (writes artifacts to disk and prints metrics)
    Why this contract matters for reliable ML delivery:
    - A single entry point makes the pipeline easy to run, automate, and test
    """
    print("[main.main] Starting pipeline")  # TODO: replace with logging later

    # --- 1) LOAD ---
    print("[main.main] 1) LOAD")  # TODO: replace with logging later
    df_raw = load_raw_data(RAW_DATA_PATH)

    # --- 2) CLEAN ---
    print("[main.main] 2) CLEAN")  # TODO: replace with logging later
    df_clean = clean_dataframe(df_raw, target_column=TARGET_COLUMN)

    # --- 3) SAVE PROCESSED CSV ---
    print("[main.main] 3) SAVE PROCESSED CSV")  # TODO: replace with logging later
    save_csv(df_clean, CLEAN_DATA_PATH)

    # --- 4) VALIDATE ---
    print("[main.main] 4) VALIDATE")  # TODO: replace with logging later
    validate_dataframe(df_clean, required_columns=[TARGET_COLUMN])

    # --- 5) SPLIT ---
    print("[main.main] 5) SPLIT")  # TODO: replace with logging later
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace or extend the baseline logic here
    # Why: Split strategy depends on leakage risk, imbalance, and how the model will be used
    # Examples:
    # 1. Add stratify=y for classification imbalance
    # 2. Use time based split for time series
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # --- 6) TRAIN ---
    print("[main.main] 6) TRAIN")  # TODO: replace with logging later
    model = train_model(X_train=X_train, y_train=y_train, problem_type=PROBLEM_TYPE)

    # --- 7) SAVE MODEL ---
    print("[main.main] 7) SAVE MODEL")  # TODO: replace with logging later
    save_model(model, MODEL_PATH)

    # --- 8) EVALUATE ---
    print("[main.main] 8) EVALUATE")  # TODO: replace with logging later
    _metric_value = evaluate_model(model=model, X_test=X_test, y_test=y_test, problem_type=PROBLEM_TYPE)

    # --- 9) INFER ---
    print("[main.main] 9) INFER")  # TODO: replace with logging later
    X_infer = X_test.head(5).copy()
    df_pred = run_inference(model=model, X_infer=X_infer)

    # --- 10) SAVE PREDICTIONS ---
    print("[main.main] 10) SAVE PREDICTIONS")  # TODO: replace with logging later
    save_csv(df_pred, PREDICTIONS_PATH)

    print("[main.main] Done")  # TODO: replace with logging later
    print(f"[main.main] Wrote {CLEAN_DATA_PATH}")  # TODO: replace with logging later
    print(f"[main.main] Wrote {MODEL_PATH}")  # TODO: replace with logging later
    print(f"[main.main] Wrote {PREDICTIONS_PATH}")  # TODO: replace with logging later


if __name__ == "__main__":
    main()