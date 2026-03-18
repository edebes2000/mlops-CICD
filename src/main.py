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

import logging

import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

import wandb

from src.logger import configure_logging

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_calibration, evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import calibrate_pipeline, train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

# -----------------------------
# Config loading and validation
# -----------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration from disk

    Why this exists
    - Centralizing config loading prevents "config drift" where different modules parse YAML differently
    - Fail fast with clear messages when config.yaml is missing or malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            "config.yaml must parse into a dictionary at the top level")

    return cfg


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Enforce a required top-level config section

    Why this exists
    - This produces an actionable error tied to config.yaml structure
    """
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(
            f"config.yaml must contain a top-level '{section}' mapping")
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
        raise ValueError(
            f"config.yaml: '{key}' must be a number. Got '{value}'") from e


def require_int(section: Dict[str, Any], key: str) -> int:
    value = section.get(key)
    try:
        return int(value)
    except Exception as e:
        raise ValueError(
            f"config.yaml: '{key}' must be an integer. Got '{value}'") from e


def require_list(section: Dict[str, Any], key: str) -> List[str]:
    value = section.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"config.yaml: '{key}' must be a list. Got type={type(value)}")
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def normalize_problem_type(problem_type: Optional[str]) -> str:
    return (problem_type or "").strip().lower()


def resolve_repo_path(project_root: Path, relative_path: str) -> Path:
    """
    Resolve a config path relative to the repo root

    This makes the repo reproducible across machines because we never rely on the current working directory
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
    Split into train, validation, test

    Why it matters
    - Train is for learning
    - Validation is for decision making during development
    - Test is the final audit, used sparingly
    """
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
        raise ValueError(
            "Split sizes must satisfy: 0 < test_size, 0 < val_size, and test_size + val_size < 1"
        )

    stratify_y = y if stratify else None

    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )

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
        logger.warning(
            "Stratified split failed, falling back to non-stratified split | error=%s",
            e,
        )

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


# -----------------------------
# W&B config helpers
# -----------------------------
def _wandb_is_enabled(cfg: Dict[str, Any]) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return False
    return bool(wandb_cfg.get("enabled", False))


def _wandb_get_str(cfg: Dict[str, Any], key: str, default: str = "") -> str:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return str(value).strip() if value is not None else default


def _wandb_get_bool(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return bool(value)


def _wandb_get_int(cfg: Dict[str, Any], key: str, default: int = 0) -> int:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    try:
        return int(value)
    except Exception:
        return default


def _wandb_get_list(cfg: Dict[str, Any], key: str) -> List[str]:
    """Safely extract a list of strings, stripping whitespace and dropping empty values."""
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return []

    value = wandb_cfg.get(key, [])
    if not isinstance(value, list):
        return []

    out: List[str] = []
    for v in value:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def _log_wandb_classification_artifacts(
    cfg: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_obj,
    stage_name: str,
) -> None:
    """
    Log optional classification plots and tables to W&B.

    Why this helper exists
    - Keep main() readable while avoiding duplicated W&B code for before/after calibration
    """
    if not hasattr(model_obj, "predict_proba"):
        logger.warning(
            "Skipping W&B classification artifacts for stage '%s' because model has no predict_proba()",
            stage_name,
        )
        return

    class_names = (
        cfg.get("wandb", {}).get("class_names", None)
        if isinstance(cfg.get("wandb"), dict)
        else None
    )

    y_probas_val = model_obj.predict_proba(X_val)

    if _wandb_get_bool(cfg, "log_auc_plots", default=False):
        wandb.log(
            {
                f"plots/roc_curve_val_{stage_name}": wandb.plot.roc_curve(
                    y_true=y_val.tolist(),
                    y_probas=y_probas_val,
                    labels=class_names,
                    title=f"Validation ROC Curve ({stage_name})",
                ),
                f"plots/pr_curve_val_{stage_name}": wandb.plot.pr_curve(
                    y_true=y_val.tolist(),
                    y_probas=y_probas_val,
                    labels=class_names,
                    title=f"Validation Precision-Recall Curve ({stage_name})",
                ),
            }
        )

    if _wandb_get_bool(cfg, "log_confusion_matrix", default=False):
        y_pred_val = model_obj.predict(X_val)
        wandb.log(
            {
                f"plots/confusion_matrix_val_{stage_name}": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_val.tolist(),
                    preds=y_pred_val.tolist() if hasattr(y_pred_val, "tolist") else list(y_pred_val),
                    class_names=class_names,
                    title=f"Validation Confusion Matrix ({stage_name})",
                )
            }
        )

    if _wandb_get_bool(cfg, "log_calibration_table", default=False):
        eval_cfg = cfg.get("evaluation", {})
        calibration_bins = int(eval_cfg.get("calibration_bins", 10)) if isinstance(
            eval_cfg, dict) else 10

        y_prob_pos = y_probas_val[:, 1]
        calib_table, _ = evaluate_calibration(
            y_true=y_val,
            y_prob=y_prob_pos,
            n_bins=calibration_bins,
        )

        if not calib_table.empty:
            wandb.log(
                {f"tables/calibration_val_{stage_name}": wandb.Table(dataframe=calib_table)})


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Load .env explicitly from repo root to avoid ambiguity
    load_dotenv(dotenv_path=project_root / ".env", override=False)

    # -----------------------------
    # Load and validate config.yaml
    # -----------------------------
    cfg = load_config(project_root / "config.yaml")

    paths_cfg = require_section(cfg, "paths")
    problem_cfg = require_section(cfg, "problem")
    split_cfg = require_section(cfg, "split")
    features_cfg = require_section(cfg, "features")
    validation_cfg = require_section(cfg, "validation")
    run_cfg = require_section(cfg, "run")
    training_cfg = require_section(cfg, "training")
    logging_cfg = require_section(cfg, "logging")

    log_file_path = resolve_repo_path(
        project_root, require_str(paths_cfg, "log_file"))
    log_level = require_str(logging_cfg, "level")

    configure_logging(
        log_level=log_level,
        log_file=log_file_path,
    )

    # -----------------------------
    # Initialize W&B Experiment Tracking
    # -----------------------------
    wandb_run = None
    if _wandb_is_enabled(cfg):
        wandb_project = _wandb_get_str(cfg, "project")
        if not wandb_project:
            raise ValueError(
                "config.yaml: wandb.project must be a non-empty string when wandb.enabled is true"
            )

        wandb_name = _wandb_get_str(cfg, "name")
        wandb_job_type = _wandb_get_str(
            cfg, "job_type", default="factory-pipeline")
        wandb_group = _wandb_get_str(cfg, "group")
        wandb_notes = _wandb_get_str(cfg, "notes")
        wandb_tags = _wandb_get_list(cfg, "tags")

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name if wandb_name else None,
            job_type=wandb_job_type,
            group=wandb_group if wandb_group else None,
            notes=wandb_notes if wandb_notes else None,
            tags=wandb_tags if wandb_tags else None,
            config=cfg,
        )

        wandb_run.summary["entrypoint"] = "python -m src.main"
        wandb_run.summary["model_artifact_path"] = str(
            require_str(paths_cfg, "model_artifact"))

        logger.info(
            "Initialized W&B run | name=%s | project=%s | job_type=%s",
            wandb_run.name,
            wandb_project,
            wandb_job_type,
        )
    else:
        logger.info("W&B disabled, continuing without experiment tracking")

    try:
        logger.info("Starting pipeline")

        problem_type = normalize_problem_type(
            require_str(problem_cfg, "problem_type"))
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "config.yaml: problem.problem_type must be 'classification' or 'regression'"
            )

        target_column = require_str(problem_cfg, "target_column")
        identifier_column = require_str(problem_cfg, "identifier_column")

        # Resolve paths
        raw_data_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "raw_data"))
        processed_data_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "processed_data"))
        model_artifact_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "model_artifact"))
        inference_data_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "inference_data"))
        predictions_artifact_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "predictions_artifact"))

        # Split settings
        test_size = require_float(split_cfg, "test_size")
        val_size = require_float(split_cfg, "val_size")
        random_state = require_int(split_cfg, "random_state")

        # Feature columns
        quantile_bin_cols = require_list(features_cfg, "quantile_bin")
        categorical_onehot_cols = require_list(
            features_cfg, "categorical_onehot")
        numeric_passthrough_cols = require_list(
            features_cfg, "numeric_passthrough")
        binary_sum_cols = require_list(features_cfg, "binary_sum_cols")
        n_bins = require_int(features_cfg, "n_bins")

        configured_cols = dedupe_preserve_order(
            quantile_bin_cols + categorical_onehot_cols +
            numeric_passthrough_cols + binary_sum_cols
        )
        if not configured_cols:
            raise ValueError(
                "config.yaml: features must define at least 1 column across the feature lists"
            )

        # Validation config
        numeric_non_negative_cols = require_list(
            validation_cfg, "numeric_non_negative_cols")
        check_missing_values = bool(
            validation_cfg.get("check_missing_values", False))

        # Run config
        include_proba_if_classification = bool(
            run_cfg.get("include_proba_if_classification", True))

        # Training config for this problem type
        model_params = training_cfg.get(problem_type, {})
        if model_params is None:
            model_params = {}
        if not isinstance(model_params, dict):
            raise ValueError(
                f"config.yaml: training.{problem_type} must be a mapping")

        # -----------------------------
        # 1) Load raw training data
        # -----------------------------
        logger.info("1) LOAD raw data")
        df_raw = load_raw_data(raw_data_path)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/raw_rows": int(df_raw.shape[0]),
                    "data/raw_cols": int(df_raw.shape[1]),
                }
            )

        # -----------------------------
        # 2) Clean training data
        # -----------------------------
        logger.info("2) CLEAN training data")
        df_clean = clean_dataframe(df_raw, target_column=target_column)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/clean_rows": int(df_clean.shape[0]),
                    "data/clean_cols": int(df_clean.shape[1]),
                }
            )

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
            target_allowed_values=[
                0, 1] if problem_type == "classification" else None,
            numeric_non_negative_cols=numeric_non_negative_cols,
        )

        # -----------------------------
        # 5) Split
        # -----------------------------
        logger.info("5) SPLIT train/val/test")
        X_full = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        identifier_col = identifier_column if identifier_column in X_full.columns else None
        X_features = X_full.drop(
            columns=[identifier_col]) if identifier_col else X_full

        X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
            X_features,
            y,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify=(problem_type == "classification"),
        )

        logger.info("Split sizes | train=%s | val=%s | test=%s",
                    X_train.shape, X_val.shape, X_test.shape)

        if len(X_test) == 0:
            raise ValueError(
                "Test split is empty. Check split ratios and dataset size")

        missing_cols = sorted(set(configured_cols) - set(X_train.columns))
        if missing_cols:
            raise ValueError(
                f"Configured feature columns not found in the dataset: {missing_cols}")

        for col in quantile_bin_cols:
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric for quantile binning. Found dtype={X_train[col].dtype}"
                )

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
        # 7) Train base model
        # -----------------------------
        logger.info("7) TRAIN base model pipeline")
        base_model = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            problem_type=problem_type,
            model_params=model_params,
        )

        models_to_evaluate = [("before_calibration", base_model)]
        final_model = base_model

        # -----------------------------
        # 7.5) Optional calibration
        # -----------------------------
        calibration_enabled = bool(
            model_params.get("calibration_enabled", False))

        if problem_type == "classification" and calibration_enabled:
            logger.info("7.5) CALIBRATE classification probabilities")

            calibration_method = str(model_params.get(
                "calibration_method", "sigmoid")).strip().lower()
            calibration_cv = int(model_params.get("calibration_cv", 3))

            final_model = calibrate_pipeline(
                pipeline=base_model,
                X_train=X_train,
                y_train=y_train,
                method=calibration_method,
                cv=calibration_cv,
            )
            models_to_evaluate.append(("after_calibration", final_model))
        else:
            logger.info("7.5) Calibration skipped")

        # -----------------------------
        # 8) Evaluate on validation split
        # -----------------------------
        logger.info("8) EVALUATE on validation split")

        val_metrics_comparison: Dict[str, Dict[str, float]] = {}

        for stage_name, model_obj in models_to_evaluate:
            logger.info("Evaluating validation stage: %s", stage_name)

            val_metrics = evaluate_model(
                model=model_obj,
                X_eval=X_val,
                y_eval=y_val,
                problem_type=problem_type,
            )
            val_metrics_comparison[stage_name] = val_metrics
            logger.info("Validation metrics [%s]: %s", stage_name, val_metrics)

            if wandb_run is not None:
                wandb.log({f"metrics/val_{stage_name}/{k}": float(v)
                          for k, v in val_metrics.items()})

            if wandb_run is not None and problem_type == "classification":
                _log_wandb_classification_artifacts(
                    cfg=cfg,
                    X_val=X_val,
                    y_val=y_val,
                    model_obj=model_obj,
                    stage_name=stage_name,
                )

        if wandb_run is not None and len(val_metrics_comparison) > 1:
            comparison_df = pd.DataFrame.from_dict(
                val_metrics_comparison, orient="index").reset_index()
            comparison_df = comparison_df.rename(columns={"index": "stage"})
            wandb.log(
                {"tables/metrics_comparison_val": wandb.Table(dataframe=comparison_df)})

        # -----------------------------
        # 9) Save model artifact
        # -----------------------------
        logger.info("9) SAVE model artifact")
        save_model(final_model, model_artifact_path)

        if wandb_run is not None:
            model_artifact_name = _wandb_get_str(
                cfg, "model_artifact_name", default="model")
            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type="model",
                description="Scikit-learn pipeline or calibrated model artifact",
            )
            model_artifact.add_file(str(model_artifact_path))
            wandb.log_artifact(model_artifact)

            if _wandb_get_bool(cfg, "log_processed_data", default=False):
                data_artifact = wandb.Artifact(
                    name=f"{model_artifact_name}-processed-data",
                    type="dataset",
                    description="Processed training dataset written by the factory pipeline",
                )
                data_artifact.add_file(str(processed_data_path))
                wandb.log_artifact(data_artifact)

        # -----------------------------
        # 10) Inference on new data
        # -----------------------------
        logger.info("10) INFER on new data file")
        if not inference_data_path.exists():
            logger.error("Inference file not found at: %s",
                         inference_data_path)
            raise FileNotFoundError(
                f"Inference file not found at: {inference_data_path}")

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
        X_infer = df_infer_clean.drop(
            columns=[infer_identifier_col]) if infer_identifier_col else df_infer_clean

        include_proba = (
            problem_type == "classification") and include_proba_if_classification

        df_predictions = run_inference(
            model=final_model,
            X_infer=X_infer,
            include_proba=include_proba,
        )

        if wandb_run is not None and _wandb_get_bool(cfg, "log_predictions_table", default=False):
            n_rows = _wandb_get_int(cfg, "predictions_table_rows", default=200)
            sample_df = df_predictions.head(n_rows)
            wandb.log(
                {"tables/predictions_preview": wandb.Table(dataframe=sample_df)})

        if infer_identifier_col:
            df_predictions.insert(0, infer_identifier_col,
                                  df_infer_clean[infer_identifier_col].values)

        logger.debug("Inference preview\n%s",
                     df_predictions.head(10).to_string(index=False))

        save_csv(df_predictions, predictions_artifact_path)

        if wandb_run is not None and _wandb_get_bool(cfg, "log_predictions", default=False):
            model_artifact_name = _wandb_get_str(
                cfg, "model_artifact_name", default="model")
            pred_artifact = wandb.Artifact(
                name=f"{model_artifact_name}-predictions",
                type="predictions",
                description="Inference outputs written by the factory pipeline",
            )
            pred_artifact.add_file(str(predictions_artifact_path))
            wandb.log_artifact(pred_artifact)

        logger.info("Done")
        logger.info("Wrote processed data: %s", processed_data_path)
        logger.info("Wrote model artifact: %s", model_artifact_path)
        logger.info("Wrote predictions: %s", predictions_artifact_path)

    except Exception:
        logger.exception("Pipeline failed")
        if wandb_run is not None:
            wandb.finish(exit_code=1)
        raise

    finally:
        if wandb_run is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
