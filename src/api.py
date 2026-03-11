# src/api.py
"""
Educational Goal:
- Turn the trained ML pipeline into a minimal web product using FastAPI
- Use Pydantic to enforce a strict data contract (types and required fields)
- Reuse existing pipeline modules to prove modularity value and prevent training serving skew

Key principle:
- No new ML logic in this file
- This file only does HTTP, schema validation, and routing to existing pipeline functions

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.clean_data import clean_dataframe
from src.infer import run_inference
from src.validate import validate_dataframe

# Keep this as the only coupling to main.py
from src.main import load_config, require_section, require_str, resolve_repo_path  # type: ignore


logger = logging.getLogger("mlops.api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

load_dotenv()


# -------------------------------------------------------------------
# 1) Pydantic Schemas (strict API contract)
# -------------------------------------------------------------------
class PatientRecord(BaseModel):
    """
    This is the API contract
    FastAPI + Pydantic will reject requests with missing fields or wrong types

    Extra fields are forbidden to avoid silently accepting junk payloads

    """

    model_config = ConfigDict(extra="forbid")

    ID: str
    rx_ds: float
    A: int
    B: int
    C: int
    D: int
    E: int
    F: int
    H: int
    I: int
    J: int
    K: int
    L: int
    M: int
    N: int
    R: int
    S: int
    T: int
    Low_inc: int
    SURG: int


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[PatientRecord]


class PredictionItem(BaseModel):
    ID: str
    prediction: int
    probability: Optional[float] = None


class PredictResponse(BaseModel):
    model_version: str
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


# -------------------------------------------------------------------
# 2) Small local helpers (avoid relying on extra helpers in main.py)
# -------------------------------------------------------------------
def _require_list(cfg: Dict[str, Any], key: str) -> List[Any]:
    """
    Fetch a list from config or fail fast with a clean error

    We keep this local to avoid assuming extra helpers exist in src.main.py

    """
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key}")
    value = cfg.get(key)
    if not isinstance(value, list):
        raise ValueError(
            f"Config key '{key}' must be a list, got {type(value).__name__}")
    return value


def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _configured_feature_columns(cfg: Dict[str, Any]) -> List[str]:
    """
    Build the required feature column list from config.yaml features section

    This removes the need for a second required_columns list under an api section

    """
    features_cfg = require_section(cfg, "features")

    quantile_bin_cols = _require_list(features_cfg, "quantile_bin")
    categorical_onehot_cols = _require_list(features_cfg, "categorical_onehot")
    numeric_passthrough_cols = _require_list(
        features_cfg, "numeric_passthrough")
    binary_sum_cols = _require_list(features_cfg, "binary_sum_cols")

    cols = _dedupe_preserve_order(
        list(quantile_bin_cols)
        + list(categorical_onehot_cols)
        + list(numeric_passthrough_cols)
        + list(binary_sum_cols)
    )

    # Defensive type check for teaching clarity
    cols_str: List[str] = []
    for c in cols:
        if not isinstance(c, str):
            raise ValueError(
                f"Feature column names must be strings, got {type(c).__name__}")
        cols_str.append(c)

    return cols_str


# -------------------------------------------------------------------
# 3) App + global state (beginner friendly, no classes)
# -------------------------------------------------------------------
app = FastAPI(title="Opioid Risk Predictor API", version="1.0.0")

GLOBAL_CONFIG: Dict[str, Any] = {}
MODEL_PIPELINE: Any = None
MODEL_VERSION: str = "unloaded"


@app.on_event("startup")
def load_model_on_startup() -> None:
    """
    Load config and model exactly once when the server starts

    We do not crash if the model is missing
    This lets /health report the failure cleanly in demos

    """
    global GLOBAL_CONFIG, MODEL_PIPELINE, MODEL_VERSION

    try:
        project_root = Path(__file__).resolve().parents[1]
        GLOBAL_CONFIG = load_config(project_root / "config.yaml")

        paths_cfg = require_section(GLOBAL_CONFIG, "paths")
        model_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "model_artifact"))

        if not model_path.exists():
            logger.error("Model file missing at %s", model_path)
            MODEL_PIPELINE = None
            MODEL_VERSION = "missing"
            return

        MODEL_PIPELINE = joblib.load(model_path)
        MODEL_VERSION = model_path.name
        logger.info("Startup complete, model loaded from %s", model_path)

    except Exception as e:
        logger.exception("Startup failed: %s", str(e))
        MODEL_PIPELINE = None
        MODEL_VERSION = "startup_error"


# -------------------------------------------------------------------
# 4) Endpoints
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Use /health or /docs or /predict"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok" if MODEL_PIPELINE is not None else "model_not_loaded",
        model_loaded=MODEL_PIPELINE is not None,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Core inference flow demonstrating reuse of batch pipeline logic

    Steps
    - Convert validated Pydantic objects to a DataFrame
    - Clean using clean_dataframe from Phase 1
    - Validate using validate_dataframe from Phase 1
    - Predict using run_inference from Phase 1
    - Return structured response

    """
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded, run `python -m src.main` first")

    try:
        records_dicts = [r.model_dump() for r in req.records]
        df_raw = pd.DataFrame(records_dicts)

        df_clean = clean_dataframe(df_raw, target_column=None)

        configured_feature_cols = _configured_feature_columns(GLOBAL_CONFIG)

        validation_cfg = require_section(GLOBAL_CONFIG, "validation")
        numeric_non_negative_cols = validation_cfg.get(
            "numeric_non_negative_cols", [])
        if not isinstance(numeric_non_negative_cols, list):
            raise ValueError(
                "validation.numeric_non_negative_cols must be a list")

        validate_dataframe(
            df=df_clean,
            required_columns=configured_feature_cols,
            check_missing_values=bool(
                validation_cfg.get("check_missing_values", False)),
            target_column=None,
            target_allowed_values=None,
            numeric_non_negative_cols=numeric_non_negative_cols,
        )

        problem_cfg = require_section(GLOBAL_CONFIG, "problem")
        identifier_col = require_str(problem_cfg, "identifier_column")

        ids = (
            df_clean[identifier_col].astype(str).tolist()
            if identifier_col in df_clean.columns
            else [str(i) for i in range(len(df_clean))]
        )

        X_infer = df_clean.drop(
            columns=[identifier_col]) if identifier_col in df_clean.columns else df_clean

        run_cfg = require_section(GLOBAL_CONFIG, "run")
        include_proba = bool(run_cfg.get(
            "include_proba_if_classification", True))

        try:
            df_pred = run_inference(
                model=MODEL_PIPELINE, X_infer=X_infer, include_proba=include_proba)
        except TypeError:
            df_pred = run_inference(model=MODEL_PIPELINE, X_infer=X_infer)

        preds: List[PredictionItem] = []
        for i in range(len(ids)):
            pred_val = int(df_pred.iloc[i]["prediction"])
            prob_val: Optional[float] = None
            if "probability" in df_pred.columns:
                prob_val = float(df_pred.iloc[i]["probability"])

            preds.append(PredictionItem(
                ID=ids[i], prediction=pred_val, probability=prob_val))

        return PredictResponse(model_version=MODEL_VERSION, predictions=preds)

    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Internal Server Error") from e
