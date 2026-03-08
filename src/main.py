# src/main.py
"""
Educational goal
- Show how an MLOps repo separates orchestration (main) from implementation (src modules)
- Show how a system becomes configurable without editing code by moving runtime settings into config.yaml
- Establish a single "golden path" entrypoint that can later be wrapped by CI and deployment

What this module owns
- Load and validate config.yaml
- Resolve repo-relative paths
- Orchestrate: load -> clean -> validate -> split -> feature recipe -> train -> evaluate -> save -> infer

What this module does not own
- Data cleaning rules (src/clean_data.py)
- Feature engineering implementations (src/features.py)
- Training algorithm implementation (src/train.py)
- Evaluation metrics (src/evaluate.py)
- Inference logic (src/infer.py)
- Validation rules (src/validate.py)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import logging
from src.logger import configure_logging

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

# -----------------------------
# Config loading and validation
# -----------------------------
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration from disk.

    Why this exists
    - Centralizing config loading prevents "config drift" where different modules parse YAML differently
    - Fail fast with clear messages when config.yaml is missing or malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse into a dictionary at the top level")

    return cfg


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Enforce a required top-level config section.

    Why this exists
    - This produces an actionable error tied to config.yaml structure
    """
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"config.yaml must contain a top-level '{section}' mapping")
    return value


def require_str(section: Dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.yaml: '{key}' must be a non-empty string")
    return value.strip()


def require_float(section: Dict[str, Any], key: str) -> float:
    value = section.get(key)
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"config.yaml: '{key}' must be a number. Got '{value}'") from e


def require_int(section: Dict[str, Any], key: str) -> int:
    value = section.get(key)
    try:
        return int(value)
    except Exception as e:
        raise ValueError(f"config.yaml: '{key}' must be an integer. Got '{value}'") from e


def require_list(section: Dict[str, Any], key: str) -> List[str]:
    value = section.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"config.yaml: '{key}' must be a list. Got type={type(value)}")
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def normalize_problem_type(problem_type: Optional[str]) -> str:
    return (problem_type or "").strip().lower()


def resolve_repo_path(project_root: Path, relative_path: str) -> Path:
    """
    Resolve a config path relative to the repo root.

    This makes the repo reproducible across machines because we never rely on the current working directory.
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("config.yaml: path values must be non-empty strings")
    return project_root / relative_path.strip()


def dedupe_preserve_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


# -----------------------------
# Data splitting
# -----------------------------
def three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split into train, validation, test.

    Why it matters
    - Train is for learning
    - Validation is for decision making during development
    - Test is the final audit, used sparingly
    """
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
        raise ValueError("Split sizes must satisfy: 0 < test_size, 0 < val_size, and test_size + val_size < 1")

    stratify_y = y if stratify else None

    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
        )

        relative_val_size = val_size / (1.0 - test_size)
        stratify_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=stratify_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except ValueError as e:
        logger.warning("Stratified split failed, falling back to non-stratified split | error=%s", e)

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    # -----------------------------
    # Load and validate config.yaml
    # -----------------------------
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config.yaml")

    paths_cfg = require_section(cfg, "paths")
    problem_cfg = require_section(cfg, "problem")
    split_cfg = require_section(cfg, "split")
    features_cfg = require_section(cfg, "features")
    validation_cfg = require_section(cfg, "validation")
    run_cfg = require_section(cfg, "run")
    training_cfg = require_section(cfg, "training")
    logging_cfg = require_section(cfg, "logging")

    log_file_path = resolve_repo_path(project_root, require_str(paths_cfg, "log_file"))
    log_level = require_str(logging_cfg, "level")

    configure_logging(
        log_level=log_level,
        log_file=log_file_path,
    )

    try:
        logger.info("Starting pipeline")

        problem_type = normalize_problem_type(require_str(problem_cfg, "problem_type"))
        if problem_type not in {"classification", "regression"}:
            raise ValueError("config.yaml: problem.problem_type must be 'classification' or 'regression'")

        target_column = require_str(problem_cfg, "target_column")
        identifier_column = require_str(problem_cfg, "identifier_column")

        # Resolve paths
        raw_data_path = resolve_repo_path(project_root, require_str(paths_cfg, "raw_data"))
        processed_data_path = resolve_repo_path(project_root, require_str(paths_cfg, "processed_data"))
        model_artifact_path = resolve_repo_path(project_root, require_str(paths_cfg, "model_artifact"))
        inference_data_path = resolve_repo_path(project_root, require_str(paths_cfg, "inference_data"))
        predictions_artifact_path = resolve_repo_path(project_root, require_str(paths_cfg, "predictions_artifact"))

        # Split settings
        test_size = require_float(split_cfg, "test_size")
        val_size = require_float(split_cfg, "val_size")
        random_state = require_int(split_cfg, "random_state")

        # Feature columns
        quantile_bin_cols = require_list(features_cfg, "quantile_bin")
        categorical_onehot_cols = require_list(features_cfg, "categorical_onehot")
        numeric_passthrough_cols = require_list(features_cfg, "numeric_passthrough")
        binary_sum_cols = require_list(features_cfg, "binary_sum_cols")
        n_bins = require_int(features_cfg, "n_bins")

        configured_cols = dedupe_preserve_order(
            quantile_bin_cols + categorical_onehot_cols + numeric_passthrough_cols + binary_sum_cols
        )
        if not configured_cols:
            raise ValueError("config.yaml: features must define at least 1 column across the feature lists")

        # Validation config
        numeric_non_negative_cols = require_list(validation_cfg, "numeric_non_negative_cols")
        check_missing_values = bool(validation_cfg.get("check_missing_values", False))

        # Run config
        include_proba_if_classification = bool(run_cfg.get("include_proba_if_classification", True))

        # Training config for this problem type
        model_params = training_cfg.get(problem_type, {})
        if model_params is None:
            model_params = {}
        if not isinstance(model_params, dict):
            raise ValueError(f"config.yaml: training.{problem_type} must be a mapping")

        # -----------------------------
        # 1) Load raw training data
        # -----------------------------
        logger.info("1) LOAD raw data")
        df_raw = load_raw_data(raw_data_path)

        # -----------------------------
        # 2) Clean training data
        # -----------------------------
        logger.info("2) CLEAN training data")
        df_clean = clean_dataframe(df_raw, target_column=target_column)

        # -----------------------------
        # 3) Save processed data
        # -----------------------------
        logger.info("3) SAVE processed data")
        save_csv(df_clean, processed_data_path)

        # -----------------------------
        # 4) Validate training data
        # -----------------------------
        logger.info("4) VALIDATE training data")
        required_columns = [target_column] + configured_cols
        validate_dataframe(
            df=df_clean,
            required_columns=required_columns,
            check_missing_values=check_missing_values,
            target_column=target_column,
            target_allowed_values=[0, 1] if problem_type == "classification" else None,
            numeric_non_negative_cols=numeric_non_negative_cols,
        )

        # -----------------------------
        # 5) Split
        # -----------------------------
        logger.info("5) SPLIT train/val/test")
        X_full = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        identifier_col = identifier_column if identifier_column in X_full.columns else None
        X_features = X_full.drop(columns=[identifier_col]) if identifier_col else X_full

        X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
            X_features,
            y,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify=(problem_type == "classification"),
        )

        logger.info("Split sizes | train=%s | val=%s | test=%s", X_train.shape, X_val.shape, X_test.shape)

        if len(X_test) == 0:
            raise ValueError("Test split is empty. Check split ratios and dataset size")

        missing_cols = sorted(set(configured_cols) - set(X_train.columns))
        if missing_cols:
            raise ValueError(f"Configured feature columns not found in the dataset: {missing_cols}")

        for col in quantile_bin_cols:
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                raise ValueError(f"Column '{col}' must be numeric for quantile binning. Found dtype={X_train[col].dtype}")

        # -----------------------------
        # 6) Build feature preprocessor
        # -----------------------------
        logger.info("6) BUILD feature recipe")
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=quantile_bin_cols,
            categorical_onehot_cols=categorical_onehot_cols,
            numeric_passthrough_cols=numeric_passthrough_cols,
            binary_sum_cols=binary_sum_cols,
            n_bins=n_bins,
        )

        # -----------------------------
        # 7) Train
        # -----------------------------
        logger.info("7) TRAIN model pipeline")
        model_pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            problem_type=problem_type,
            model_params=model_params,
        )

        # -----------------------------
        # 8) Evaluate (validation)
        # -----------------------------
        logger.info("8) EVALUATE on validation split")
        val_metrics = evaluate_model(
            model=model_pipeline,
            X_eval=X_val,
            y_eval=y_val,
            problem_type=problem_type,
        )
        logger.info("Validation metrics: %s", val_metrics)

        # -----------------------------
        # 9) Save model artifact
        # -----------------------------
        logger.info("9) SAVE model artifact")
        save_model(model_pipeline, model_artifact_path)

        # -----------------------------
        # 10) Inference on new data
        # -----------------------------
        logger.info("10) INFER on new data file")
        if not inference_data_path.exists():
            logger.error("Inference file not found at: %s", inference_data_path)
            raise FileNotFoundError(f"Inference file not found at: {inference_data_path}")

        df_infer_raw = load_raw_data(inference_data_path)
        df_infer_clean = clean_dataframe(df_infer_raw, target_column=None)

        validate_dataframe(
            df=df_infer_clean,
            required_columns=configured_cols,
            check_missing_values=check_missing_values,
            target_column=None,
            target_allowed_values=None,
            numeric_non_negative_cols=numeric_non_negative_cols,
        )

        infer_identifier_col = identifier_column if identifier_column in df_infer_clean.columns else None
        X_infer = df_infer_clean.drop(columns=[infer_identifier_col]) if infer_identifier_col else df_infer_clean

        include_proba = (problem_type == "classification") and include_proba_if_classification

        df_predictions = run_inference(model=model_pipeline, X_infer=X_infer, include_proba=include_proba)

        if infer_identifier_col:
            df_predictions.insert(0, infer_identifier_col, df_infer_clean[infer_identifier_col].values)

        logger.debug("Inference preview\n%s", df_predictions.head(10).to_string(index=False))

        save_csv(df_predictions, predictions_artifact_path)

        logger.info("Done")
        logger.info("Wrote processed data: %s", processed_data_path)
        logger.info("Wrote model artifact: %s", model_artifact_path)
        logger.info("Wrote predictions: %s", predictions_artifact_path)

    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
