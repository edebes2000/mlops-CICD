# tests/test_main.py
"""
Educational Goal:
- Why these tests exist in an MLOps system: Verify that the orchestrator is mathematically correct (splits) and operationally correct (creates artifacts).
- Responsibility: Unit test the split logic, run a safe end-to-end pipeline in an isolated temp directory, and enforce separation of concerns.
"""

import copy
from pathlib import Path

import joblib
import pandas as pd
import pytest

import src.main as main_module


# --------------------------------------------------------
# 1) UNIT TESTS: 3-WAY SPLIT MATH
# --------------------------------------------------------
@pytest.fixture
def dummy_split_data():
    """Predictable dataset for split testing (100 rows, imbalanced target)."""
    X = pd.DataFrame({"feature1": range(100), "feature2": range(100)})
    y = pd.Series([0] * 80 + [1] * 20, name="target")
    return X, y


def test_three_way_split_correct_sizes(dummy_split_data):
    """Ensure the math correctly partitions a 100-row dataset."""
    X, y = dummy_split_data

    X_train, X_val, X_test, y_train, y_val, y_test = main_module._three_way_split(
        X, y, test_size=0.10, val_size=0.20, random_state=42, stratify=False
    )

    assert len(X_test) == 10
    assert len(X_val) == 20
    assert len(X_train) == 70


def test_three_way_split_stratification_preserves_ratio(dummy_split_data):
    """Ensure classification splits preserve the 80/20 class imbalance."""
    X, y = dummy_split_data

    _, _, _, _, _, y_test = main_module._three_way_split(
        X, y, test_size=0.10, val_size=0.20, random_state=42, stratify=True
    )

    # With 80/20 overall and 10 test rows, we expect exactly 2 positive labels in the test set
    assert (y_test == 1).sum() == 2


def test_three_way_split_invalid_sizes_raises(dummy_split_data):
    """Crash if configuration asks for impossible split ratios."""
    X, y = dummy_split_data

    with pytest.raises(ValueError, match="split sizes must satisfy"):
        main_module._three_way_split(
            X, y, test_size=0.60, val_size=0.50, random_state=42, stratify=False
        )


# --------------------------------------------------------
# 2) END-TO-END ORCHESTRATION TESTS
# --------------------------------------------------------
def _make_synthetic_raw_df(n_rows: int = 200) -> pd.DataFrame:
    """Builds a minimal dataset perfectly matching SETTINGS expectations."""
    df = pd.DataFrame({
        "ID": list(range(n_rows)),
        "OD": [i % 2 for i in range(n_rows)],
        "rx_ds": list(range(n_rows)),
    })
    for col in main_module.BINARY_SUM_COLS:
        df[col] = [(i % 2) for i in range(n_rows)]
    return df


def _patch_paths_and_settings(monkeypatch, tmp_path: Path):
    """
    Redirects all I/O paths to tmp_path.
    Uses deepcopy on SETTINGS to prevent cross-test state leakage.
    """
    raw_path = tmp_path / "data" / "raw" / "opiod_raw_data.csv"
    clean_path = tmp_path / "data" / "processed" / "clean.csv"
    model_path = tmp_path / "models" / "model.joblib"
    predictions_path = tmp_path / "reports" / "predictions.csv"

    monkeypatch.setattr(main_module, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(main_module, "CLEAN_DATA_PATH", clean_path)
    monkeypatch.setattr(main_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(main_module, "PREDICTIONS_PATH",
                        predictions_path)

    patched_settings = copy.deepcopy(main_module.SETTINGS)
    patched_settings["is_example_config"] = False
    monkeypatch.setattr(main_module, "SETTINGS", patched_settings)

    return raw_path, clean_path, model_path, predictions_path


def test_main_end_to_end_creates_clean_and_model_artifacts(tmp_path, monkeypatch):
    """Proves the pipeline successfully cleans data, trains a model, and saves artifacts."""
    # Unpack 4 paths now
    raw_path, clean_path, model_path, predictions_path = _patch_paths_and_settings(
        monkeypatch, tmp_path)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_raw_df(n_rows=200).to_csv(raw_path, index=False)

    main_module.main()

    # Verify Data Artifact
    assert clean_path.exists()
    df_clean = pd.read_csv(clean_path)
    assert "ID" not in df_clean.columns
    assert "OD" in df_clean.columns

    # Verify Model Artifact
    assert model_path.exists()
    model = joblib.load(model_path)
    assert hasattr(model, "predict")

    # Verify Predictions Artifact
    assert predictions_path.exists()
    df_preds = pd.read_csv(predictions_path)
    assert "prediction" in df_preds.columns
    assert "proba" in df_preds.columns  # Because it's a classification problem
    assert len(df_preds) == 10  # Because we sample min(10, len(X_test))


def test_main_calls_evaluate_and_infer_modules(tmp_path, monkeypatch):
    """
    Educational Note: This is a "Spy Test".
    We intercept the function calls to ensure the orchestrator actually delegates
    work to the correct modules, proving our separation of concerns works.
    """
    raw_path, _, _, predictions_path = _patch_paths_and_settings(
        monkeypatch, tmp_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_raw_df().to_csv(raw_path, index=False)

    called = {"evaluate": 0, "infer": 0}
    original_evaluate = main_module.evaluate_model
    original_infer = main_module.run_inference

    # Define our spies
    def _spy_evaluate(*args, **kwargs):
        called["evaluate"] += 1
        return original_evaluate(*args, **kwargs)

    def _spy_infer(*args, **kwargs):
        called["infer"] += 1
        return original_infer(*args, **kwargs)

    # Inject our spies into the orchestrator
    monkeypatch.setattr(main_module, "evaluate_model", _spy_evaluate)
    monkeypatch.setattr(main_module, "run_inference", _spy_infer)

    main_module.main()

    # Enforce the contract: The modules MUST have been used
    assert called["evaluate"] == 1
    assert called["infer"] == 1


def test_main_raises_when_raw_data_missing(tmp_path, monkeypatch):
    """Pipeline must crash immediately if input data is missing."""
    raw_path, _, _, predictions_path = _patch_paths_and_settings(
        monkeypatch, tmp_path)
    assert not raw_path.exists()

    with pytest.raises(FileNotFoundError):
        main_module.main()


def test_main_raises_when_example_config_enabled(tmp_path, monkeypatch):
    """Pipeline must crash if the instructor example block is active."""
    _patch_paths_and_settings(monkeypatch, tmp_path)

    patched_settings = copy.deepcopy(main_module.SETTINGS)
    patched_settings["is_example_config"] = True
    monkeypatch.setattr(main_module, "SETTINGS", patched_settings)

    with pytest.raises(ValueError, match="SETTINGS is an example"):
        main_module.main()
