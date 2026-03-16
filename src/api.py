# src/api.py
"""
Educational Goal:
- Turn the trained ML pipeline into a minimal web product using FastAPI.
- Use Pydantic to enforce a strict data contract (types and required fields).
- Reuse existing pipeline modules to prove modularity value and prevent "Training-Serving Skew".
- (Phase 4 Additions) Implement Scalable Observability: System Logs (Layer 1) and Async Batched ML Logs (Layer 2).

Key principle:
- NO NEW ML LOGIC IN THIS FILE.
- This file only does HTTP, schema validation, startup loading, observability logging, and routing.
"""

from contextlib import asynccontextmanager
import logging
import os
import time
import uuid
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
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
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "ID": "P001",
                "rx_ds": 12.0,
                "A": 1, "B": 0, "C": 1, "D": 0, "E": 1, "F": 0,
                "H": 0, "I": 1, "J": 0, "K": 0, "L": 1, "M": 0,
                "N": 0, "R": 1, "S": 0, "T": 1, "Low_inc": 1, "SURG": 0,
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
def _require_list(cfg: Dict[str, Any], key: str) -> List[Any]:
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
    features_cfg = require_section(cfg, "features")
    cols = _dedupe_preserve_order(
        list(_require_list(features_cfg, "quantile_bin"))
        + list(_require_list(features_cfg, "categorical_onehot"))
        + list(_require_list(features_cfg, "numeric_passthrough"))
        + list(_require_list(features_cfg, "binary_sum_cols"))
    )
    return [str(c) for c in cols]


# -------------------------------------------------------------------
# 3) Lifespan: Load shared resources once at API startup
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        project_root = Path(__file__).resolve().parents[1]
        app.state.global_config = load_config(project_root / "config.yaml")
        paths_cfg = require_section(app.state.global_config, "paths")

        model_source = os.getenv("MODEL_SOURCE", "local").lower()

        if model_source == "wandb":
            logger.info(
                "MODEL_SOURCE=wandb → fetching model from W&B Registry")
            wandb_entity = os.getenv("WANDB_ENTITY")
            # CRITICAL MLOPS FIX: Default to 'prod' instead of 'latest'
            artifact_alias = os.getenv("WANDB_MODEL_ALIAS", "prod")

            wandb_cfg = app.state.global_config.get("wandb", {})
            wandb_project = wandb_cfg.get("project")
            artifact_name = wandb_cfg.get("model_artifact_name")

            if not wandb_entity or not wandb_project or not artifact_name:
                raise ValueError(
                    "Missing required W&B credentials or config settings.")

            artifact_path = f"{wandb_entity}/{wandb_project}/{artifact_name}:{artifact_alias}"
            wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)

            api = wandb.Api()
            artifact = api.artifact(artifact_path)
            artifact_dir = artifact.download()
            model_path = Path(artifact_dir) / "model.joblib"
            logger.info("Downloaded model from W&B: %s", artifact_path)

        else:
            logger.info("MODEL_SOURCE=local → using local model artifact")
            model_path = resolve_repo_path(
                project_root, require_str(paths_cfg, "model_artifact"))

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


app = FastAPI(title="Opioid Risk Predictor API",
              version="1.0.0", lifespan=lifespan)


# -------------------------------------------------------------------
# 4) Observability Architecture (Layer 1 & Layer 2)
# -------------------------------------------------------------------

# --- LAYER 1: System Monitoring Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    logger.info(
        f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Latency: {latency:.4f}s")
    return response


# --- LAYER 2: ML Monitoring Buffer ---
LOG_BUFFER = []
# Protects our buffer from race conditions during concurrent API requests
BUFFER_LOCK = Lock()
BATCH_SIZE = 10


def flush_logs_to_wandb(batch_data: list, project_name: str):
    """Ephemeral W&B run to securely log the batch as a Table."""
    # CRITICAL MLOPS FIX: Do not attempt to hit W&B if disabled (e.g., during CI tests)
    if os.getenv("WANDB_MODE", "").lower() == "disabled":
        logger.info("Skipping W&B flush because WANDB_MODE=disabled")
        return

    try:
        run = wandb.init(project=project_name,
                         job_type="inference-batch", reinit=True)
        feature_keys = list(batch_data[0]['features'].keys())
        # Added 'latency' to correlate model speed with data payloads
        columns = ["req_id", "timestamp", "model_version",
                   "latency", "prediction", "probability"] + feature_keys

        table = wandb.Table(columns=columns)
        for item in batch_data:
            row = [
                item['req_id'], item['timestamp'], item['model_version'], item['latency'],
                item['prediction'], item['probability']
            ] + [item['features'].get(k) for k in feature_keys]
            table.add_data(*row)

        run.log({"inference_logs": table})
        run.finish()
        logger.info(f"Flushed {len(batch_data)} ML logs to W&B.")
    except Exception as e:
        logger.error(f"Failed to flush logs to W&B: {e}")


# -------------------------------------------------------------------
# 5) Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Use /health or /docs to test the API"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_loaded = getattr(app.state, "model_pipeline", None) is not None
    model_version = getattr(app.state, "model_version", "unloaded")
    return HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_version=model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    model_pipeline = getattr(app.state, "model_pipeline", None)
    global_config = getattr(app.state, "global_config", {})
    model_version = getattr(app.state, "model_version", "unloaded")

    if model_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded, run `python -m src.main` first")

    try:
        # Start timing the inference specifically for our ML Logs
        inference_start_time = time.time()

        records_dicts = [r.model_dump() for r in req.records]
        df_raw = pd.DataFrame(records_dicts)
        df_clean = clean_dataframe(df_raw, target_column=None)

        configured_feature_cols = _configured_feature_columns(global_config)
        validation_cfg = require_section(global_config, "validation")
        numeric_non_negative_cols = validation_cfg.get(
            "numeric_non_negative_cols", [])

        validate_dataframe(
            df=df_clean,
            required_columns=configured_feature_cols,
            check_missing_values=bool(
                validation_cfg.get("check_missing_values", False)),
            target_column=None,
            target_allowed_values=None,
            numeric_non_negative_cols=numeric_non_negative_cols,
        )

        problem_cfg = require_section(global_config, "problem")
        identifier_col = require_str(problem_cfg, "identifier_column")

        ids = (
            df_clean[identifier_col].astype(str).tolist()
            if identifier_col in df_clean.columns
            else [str(i) for i in range(len(df_clean))]
        )
        X_infer = df_clean.drop(
            columns=[identifier_col]) if identifier_col in df_clean.columns else df_clean

        run_cfg = require_section(global_config, "run")
        include_proba = bool(run_cfg.get(
            "include_proba_if_classification", True))

        try:
            df_pred = run_inference(
                model=model_pipeline, X_infer=X_infer, include_proba=include_proba)
        except TypeError:
            df_pred = run_inference(model=model_pipeline, X_infer=X_infer)

        inference_latency = time.time() - inference_start_time
        current_time = time.time()
        preds: List[PredictionItem] = []

        # --- THREAD-SAFE OBSERVABILITY INJECTION ---
        with BUFFER_LOCK:
            for i in range(len(ids)):
                pred_val = int(df_pred.iloc[i]["prediction"])
                prob_val: Optional[float] = None

                if "proba" in df_pred.columns:
                    prob_val = float(df_pred.iloc[i]["proba"])

                preds.append(
                    PredictionItem(
                        ID=ids[i], prediction=pred_val, probability=prob_val)
                )

                LOG_BUFFER.append({
                    "req_id": ids[i],
                    "timestamp": current_time,
                    "model_version": model_version,
                    "latency": inference_latency,
                    "prediction": pred_val,
                    "probability": prob_val,
                    "features": records_dicts[i]
                })

            if len(LOG_BUFFER) >= BATCH_SIZE:
                batch_copy = LOG_BUFFER.copy()
                LOG_BUFFER.clear()
                wandb_project = global_config.get("wandb", {}).get(
                    "project", "opioid-risk-classification")
                background_tasks.add_task(
                    flush_logs_to_wandb, batch_copy, wandb_project)
        # -------------------------------------------

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
            status_code=500, detail="Internal Server Error") from e
