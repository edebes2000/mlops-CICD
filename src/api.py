# src/api.py
"""
Educational Goal:
- Turn the trained ML pipeline into a minimal web product using FastAPI.
- Use Pydantic to enforce a strict data contract (types and required fields).
- Reuse existing pipeline modules to prove modularity value and prevent "Training-Serving Skew".

Key principle:
- NO NEW ML LOGIC IN THIS FILE.
- This file only does HTTP, schema validation, startup loading, and routing to existing pipeline functions.
"""

from contextlib import asynccontextmanager
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

# Keep this as the only coupling to main.py to keep the API decoupled
from src.main import load_config, require_section, require_str, resolve_repo_path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Load environment variables from .env (e.g. for config paths or secrets)
load_dotenv()


# -------------------------------------------------------------------
# 1) Pydantic Schemas (The "Bouncer" / Strict API Contract)
# -------------------------------------------------------------------
class PatientRecord(BaseModel):
    """
    This is the API contract. FastAPI + Pydantic will automatically reject 
    requests with missing fields or wrong data types before they reach our ML code.

    Why `extra="forbid"`?
    In standard web development, extra JSON fields are often ignored. 
    In MLOps, silently accepting extra or misspelled features is dangerous 
    and leads to silent pipeline failures. We force upstream systems to be exact.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "ID": "P001",
                "rx_ds": 12,
                "A": 1,
                "B": 0,
                "C": 1,
                "D": 0,
                "E": 1,
                "F": 0,
                "H": 0,
                "I": 1,
                "J": 0,
                "K": 0,
                "L": 1,
                "M": 0,
                "N": 0,
                "R": 1,
                "S": 0,
                "T": 1,
                "Low_inc": 1,
                "SURG": 0,
            }
        },
    )

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
# 2) Small Local Helpers
# -------------------------------------------------------------------
# We build these helpers here instead of relying on main.py.
# This ensures that if a student breaks main.py, the API can still stand up independently.

def _require_list(cfg: Dict[str, Any], key: str) -> List[Any]:
    """Fetch a list from config or fail fast with a clean error."""
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key}")

    value = cfg.get(key)
    if not isinstance(value, list):
        raise ValueError(
            f"Config key '{key}' must be a list, got {type(value).__name__}"
        )
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
    """Build the required feature column list from config.yaml features section."""
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

    cols_str: List[str] = []
    for c in cols:
        if not isinstance(c, str):
            raise ValueError(
                f"Feature column names must be strings, got {type(c).__name__}"
            )
        cols_str.append(c)

    return cols_str


# -------------------------------------------------------------------
# 3) Lifespan: Load shared resources once at API startup
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load config and model exactly once when the server starts.

    Why this exists (Latency & Stability)
    - If we load the model from disk inside `/predict`, every user waits ~200ms.
      By loading it here, it sits in RAM, and predictions take ~2ms.
    - If the model is missing, we DO NOT crash. We log an error and set state to None.
      This allows the `/health` endpoint to start up and tell the user exactly what is wrong.
    """
    try:
        project_root = Path(__file__).resolve().parents[1]
        app.state.global_config = load_config(project_root / "config.yaml")

        paths_cfg = require_section(app.state.global_config, "paths")
        model_path = resolve_repo_path(
            project_root,
            require_str(paths_cfg, "model_artifact"),
        )

        if not model_path.exists():
            logger.error("Model file missing at %s", model_path)
            app.state.model_pipeline = None
            app.state.model_version = "missing"
        else:
            app.state.model_pipeline = joblib.load(model_path)
            app.state.model_version = model_path.name
            logger.info("Startup complete, model loaded from %s", model_path)

    except Exception as e:
        logger.exception("Startup failed: %s", str(e))
        app.state.global_config = {}
        app.state.model_pipeline = None
        app.state.model_version = "startup_error"

    yield

    logger.info("API shutdown complete")

# Create FastAPI app with the above lifespan
app = FastAPI(
    title="Opioid Risk Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# 4) Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Use /health or /docs to test the API"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    # Safely retrieve from app.state
    model_loaded = getattr(app.state, "model_pipeline", None) is not None
    model_version = getattr(app.state, "model_version", "unloaded")

    return HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_version=model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Core inference flow demonstrating reuse of batch pipeline logic.

    Preventing Training-Serving Skew
    Notice how we do NOT write new Pandas logic here. We simply pass the 
    incoming data through the exact same functions (`clean_dataframe`, 
    `validate_dataframe`, `run_inference`) that we used during training.
    """
    model_pipeline = getattr(app.state, "model_pipeline", None)
    global_config = getattr(app.state, "global_config", {})
    model_version = getattr(app.state, "model_version", "unloaded")

    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded, run `python -m src.main` first",
        )

    try:
        # 1. Convert Validated Pydantic objects to a DataFrame
        records_dicts = [r.model_dump() for r in req.records]
        df_raw = pd.DataFrame(records_dicts)

        # 2. Clean Data
        df_clean = clean_dataframe(df_raw, target_column=None)

        # 3. Validate Data
        configured_feature_cols = _configured_feature_columns(global_config)
        validation_cfg = require_section(global_config, "validation")
        numeric_non_negative_cols = validation_cfg.get(
            "numeric_non_negative_cols", []
        )
        if not isinstance(numeric_non_negative_cols, list):
            raise ValueError(
                "validation.numeric_non_negative_cols must be a list"
            )

        validate_dataframe(
            df=df_clean,
            required_columns=configured_feature_cols,
            check_missing_values=bool(
                validation_cfg.get("check_missing_values", False)
            ),
            target_column=None,
            target_allowed_values=None,
            numeric_non_negative_cols=numeric_non_negative_cols,
        )

        # 4. Handle IDs
        problem_cfg = require_section(global_config, "problem")
        identifier_col = require_str(problem_cfg, "identifier_column")

        ids = (
            df_clean[identifier_col].astype(str).tolist()
            if identifier_col in df_clean.columns
            else [str(i) for i in range(len(df_clean))]
        )

        X_infer = (
            df_clean.drop(columns=[identifier_col])
            if identifier_col in df_clean.columns
            else df_clean
        )

        # 5. Predict
        run_cfg = require_section(global_config, "run")
        include_proba = bool(
            run_cfg.get("include_proba_if_classification", True)
        )

        try:
            df_pred = run_inference(
                model=model_pipeline,
                X_infer=X_infer,
                include_proba=include_proba,
            )
        except TypeError:
            df_pred = run_inference(
                model=model_pipeline,
                X_infer=X_infer,
            )

        # 6. Format API Response
        preds: List[PredictionItem] = []
        for i in range(len(ids)):
            pred_val = int(df_pred.iloc[i]["prediction"])
            prob_val: Optional[float] = None

            if "proba" in df_pred.columns:
                prob_val = float(df_pred.iloc[i]["proba"])

            preds.append(
                PredictionItem(
                    ID=ids[i],
                    prediction=pred_val,
                    probability=prob_val,
                )
            )

        return PredictResponse(
            model_version=model_version,
            predictions=preds,
        )

    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        ) from e
