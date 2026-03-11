# tests/test_main.py
"""
Educational Goal:
- Verify orchestrator correctness with a config-driven pipeline
- Unit test split math
- Run a safe end-to-end pipeline in an isolated temp directory
- Enforce separation of concerns by spying on module delegation
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import pandas as pd
import pytest

import src.main as main_module


# --------------------------------------------------------
# 1) UNIT TESTS: 3-WAY SPLIT MATH
# --------------------------------------------------------
@pytest.fixture
def dummy_split_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Predictable dataset for split testing (100 rows, imbalanced target)"""
    X = pd.DataFrame({"feature1": range(100), "feature2": range(100)})
    y = pd.Series([0] * 80 + [1] * 20, name="target")
    return X, y


def test_three_way_split_correct_sizes(dummy_split_data):
    """Ensure the math correctly partitions a 100-row dataset"""
    X, y = dummy_split_data

    X_train, X_val, X_test, y_train, y_val, y_test = main_module.three_way_split(
        X, y, test_size=0.10, val_size=0.20, random_state=42, stratify=False
    )

    assert len(X_test) == 10
    assert len(X_val) == 20
    assert len(X_train) == 70
    assert len(y_train) == 70
    assert len(y_val) == 20
    assert len(y_test) == 10


def test_three_way_split_stratification_preserves_ratio(dummy_split_data):
    """Ensure classification splits preserve the 80/20 class imbalance"""
    X, y = dummy_split_data

    _, _, _, _, _, y_test = main_module.three_way_split(
        X, y, test_size=0.10, val_size=0.20, random_state=42, stratify=True
    )

    # With 80/20 overall and 10 test rows, we expect exactly 2 positive labels in the test set
    assert (y_test == 1).sum() == 2


def test_three_way_split_invalid_sizes_raises(dummy_split_data):
    """Crash if configuration asks for impossible split ratios"""
    X, y = dummy_split_data

    with pytest.raises(ValueError, match="Split sizes must satisfy"):
        main_module.three_way_split(
            X, y, test_size=0.60, val_size=0.50, random_state=42, stratify=False
        )


# --------------------------------------------------------
# 2) END-TO-END ORCHESTRATION TESTS (CONFIG-DRIVEN)
# --------------------------------------------------------
def _binary_sum_cols_for_tests() -> List[str]:
    """
    Keep this local to the test suite so tests do not depend on internal module constants.
    Must match config.yaml used in the test.
    """
    return [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "R",
        "S",
        "T",
        "Low_inc",
        "SURG",
    ]


def _make_synthetic_raw_df(binary_sum_cols: List[str], n_rows: int = 200) -> pd.DataFrame:
    """Build a minimal dataset matching the feature config and target expectations"""
    df = pd.DataFrame(
        {
            "ID": list(range(n_rows)),
            "OD": [i % 2 for i in range(n_rows)],
            "rx_ds": list(range(n_rows)),
        }
    )
    for col in binary_sum_cols:
        df[col] = [(i % 2) for i in range(n_rows)]
    return df


def _make_synthetic_inference_df(binary_sum_cols: List[str], n_rows: int = 10) -> pd.DataFrame:
    """
    Build inference inputs that match feature expectations
    Notes
    - No OD column by design
    - Includes ID for traceability
    """
    df = pd.DataFrame(
        {
            "ID": list(range(1000, 1000 + n_rows)),
            "rx_ds": list(range(10, 10 + n_rows)),
        }
    )
    for col in binary_sum_cols:
        df[col] = [(i % 2) for i in range(n_rows)]
    return df


def _make_test_config(tmp_path: Path, binary_sum_cols: List[str]) -> Dict[str, Any]:
    """
    Build an in-memory config dict that mirrors the real config.yaml schema.
    We return a dict and monkeypatch main_module.load_config to return it.
    """
    raw_path = tmp_path / "data" / "raw" / "opiod_raw_data.csv"
    clean_path = tmp_path / "data" / "processed" / "clean.csv"
    model_path = tmp_path / "models" / "model.joblib"
    inference_path = tmp_path / "data" / "inference" / "opioid_infer_01.csv"
    predictions_path = tmp_path / "reports" / "predictions.csv"
    log_file_path = tmp_path / "logs" / "pipeline.log" # <-- ADDED THIS

    return {
        "paths": {
            # Values are strings because main resolves them relative to project root
            "raw_data": str(raw_path),
            "processed_data": str(clean_path),
            "model_artifact": str(model_path),
            "inference_data": str(inference_path),
            "predictions_artifact": str(predictions_path),
            "log_file": str(log_file_path), # <-- ADDED THIS
        },
        "problem": {"target_column": "OD", "problem_type": "classification", "identifier_column": "ID"},
        "split": {"test_size": 0.10, "val_size": 0.20, "random_state": 42},
        "features": {
            "quantile_bin": ["rx_ds"],
            "categorical_onehot": [],
            "numeric_passthrough": [],
            "binary_sum_cols": binary_sum_cols,
            "n_bins": 4,
        },
        "validation": {"numeric_non_negative_cols": ["rx_ds"], "check_missing_values": False},
        "run": {"include_proba_if_classification": True, "overwrite_outputs": True},
        "logging": {"level": "INFO", "format": "text"},
        "training": {
            "classification": {
                "model_type": "logistic_regression",
                "max_iter": 200,
                "solver": "liblinear",
                "random_state": 42,
                "class_weight": "balanced",
            },
            "regression": {"model_type": "linear_regression"},
        },
    }


def _patch_config_loader(monkeypatch, cfg: Dict[str, Any]):
    """
    Patch main_module.load_config so main() reads our in-memory config.
    This keeps the test isolated and avoids having to build a temp repo root.
    """

    def _fake_load_config(_config_path: Path) -> Dict[str, Any]:
        return cfg

    monkeypatch.setattr(main_module, "load_config", _fake_load_config)


def test_main_end_to_end_creates_clean_model_and_predictions_artifacts(tmp_path, monkeypatch):
    """Proves the pipeline cleans data, trains a model, and saves artifacts including inference outputs"""
    binary_sum_cols = _binary_sum_cols_for_tests()
    cfg = _make_test_config(tmp_path, binary_sum_cols)
    _patch_config_loader(monkeypatch, cfg)

    raw_path = Path(cfg["paths"]["raw_data"])
    inference_path = Path(cfg["paths"]["inference_data"])
    clean_path = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["model_artifact"])
    predictions_path = Path(cfg["paths"]["predictions_artifact"])

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_raw_df(binary_sum_cols=binary_sum_cols, n_rows=200).to_csv(raw_path, index=False)

    inference_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_inference_df(binary_sum_cols=binary_sum_cols, n_rows=10).to_csv(inference_path, index=False)

    main_module.main()

    # Cleaned data artifact
    assert clean_path.exists()
    df_clean = pd.read_csv(clean_path)
    assert "ID" in df_clean.columns

    # Model artifact
    assert model_path.exists()
    model = joblib.load(model_path)
    assert hasattr(model, "predict")

    # Predictions artifact
    assert predictions_path.exists()
    df_preds = pd.read_csv(predictions_path)

    assert "prediction" in df_preds.columns
    assert "proba" in df_preds.columns
    assert len(df_preds) == 10

    # Traceability contract: keep ID for audit joins
    assert "ID" in df_preds.columns
    assert df_preds["ID"].notna().all()


def test_main_calls_evaluate_and_infer_modules(tmp_path, monkeypatch):
    """
    Spy test
    We intercept calls to ensure the orchestrator delegates work to the correct modules
    """
    binary_sum_cols = _binary_sum_cols_for_tests()
    cfg = _make_test_config(tmp_path, binary_sum_cols)
    _patch_config_loader(monkeypatch, cfg)

    raw_path = Path(cfg["paths"]["raw_data"])
    inference_path = Path(cfg["paths"]["inference_data"])

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_raw_df(binary_sum_cols=binary_sum_cols, n_rows=200).to_csv(raw_path, index=False)

    inference_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_inference_df(binary_sum_cols=binary_sum_cols, n_rows=10).to_csv(inference_path, index=False)

    called = {"evaluate": 0, "infer": 0}
    original_evaluate = main_module.evaluate_model
    original_infer = main_module.run_inference

    def _spy_evaluate(*args, **kwargs):
        called["evaluate"] += 1
        return original_evaluate(*args, **kwargs)

    def _spy_infer(*args, **kwargs):
        called["infer"] += 1
        return original_infer(*args, **kwargs)

    monkeypatch.setattr(main_module, "evaluate_model", _spy_evaluate)
    monkeypatch.setattr(main_module, "run_inference", _spy_infer)

    main_module.main()

    assert called["evaluate"] == 1
    assert called["infer"] == 1


def test_main_raises_when_raw_data_missing(tmp_path, monkeypatch):
    """Pipeline must crash immediately if raw input data is missing"""
    binary_sum_cols = _binary_sum_cols_for_tests()
    cfg = _make_test_config(tmp_path, binary_sum_cols)
    _patch_config_loader(monkeypatch, cfg)

    raw_path = Path(cfg["paths"]["raw_data"])
    assert not raw_path.exists()

    with pytest.raises(FileNotFoundError):
        main_module.main()