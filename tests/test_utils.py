import pandas as pd
import pytest
from pathlib import Path

from src.utils import load_csv, save_csv, save_model, load_model


def test_save_and_load_csv_roundtrip(tmp_path: Path):
    df_in = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out_path = tmp_path / "nested" / "sample.csv"

    save_csv(df_in, out_path)
    assert out_path.exists()
    assert out_path.is_file()

    df_out = load_csv(out_path)

    # Deterministic contract: save_csv uses index=False, so index may differ on reload
    pd.testing.assert_frame_equal(df_in.reset_index(drop=True), df_out.reset_index(drop=True))


def test_load_csv_rejects_non_path_type():
    with pytest.raises(TypeError):
        load_csv("not_a_path")  # type: ignore[arg-type]


def test_load_csv_rejects_directory_path(tmp_path: Path):
    dir_path = tmp_path / "some_dir"
    dir_path.mkdir()

    with pytest.raises(ValueError, match="exists but is not a file"):
        load_csv(dir_path)


def test_load_model_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "models" / "missing.joblib"

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model(missing)


def test_save_and_load_model_roundtrip(tmp_path: Path):
    # Keep it dependency-light: a plain dict is joblib-serializable
    model_in = {"model_name": "demo", "version": 1}
    model_path = tmp_path / "models" / "model.joblib"

    save_model(model_in, model_path)
    assert model_path.exists()
    assert model_path.is_file()

    model_out = load_model(model_path)
    assert model_out == model_in