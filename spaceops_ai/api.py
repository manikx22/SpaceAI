"""FastAPI service for SpaceOps AI inference and project health."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from utils.digital_twin import compare_state, simulate_action
from utils.inference import ensure_artifacts, score_telemetry


class TelemetryPoint(BaseModel):
    """Single telemetry sample for API inference."""

    cycle: int | None = Field(default=None, ge=1)
    temperature: float = Field(ge=0.0, le=1.0)
    voltage: float = Field(ge=0.0, le=1.0)
    vibration: float = Field(ge=0.0, le=1.0)
    pressure: float = Field(ge=0.0, le=1.0)
    fuel_flow: float = Field(ge=0.0, le=1.0)
    rpm: float = Field(ge=0.0, le=1.0)


class PredictionRequest(BaseModel):
    """Batch telemetry history for model inference."""

    satellite_name: str = Field(default="SAT-API-01")
    telemetry: list[TelemetryPoint]


class SimulationRequest(PredictionRequest):
    """Payload for action simulation."""

    action: str
    operator_mode: str = Field(default="Assist")


app = FastAPI(title="SpaceOps AI API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    """Health endpoint for deployment/runtime checks."""
    artifacts = ensure_artifacts()
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "processed_rows": int(len(artifacts.df)),
        "artifacts": {
            "anomaly_model": config.ANOMALY_MODEL_PATH.exists(),
            "iforest_model": config.ANOMALY_IFOREST_PATH.exists(),
            "lstm_model": config.LSTM_MODEL_PATH.exists(),
        },
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    """Run anomaly detection, failure prediction, and recommendation inference."""
    if not request.telemetry:
        return {"error": "telemetry payload must contain at least one sample"}

    frame = pd.DataFrame([row.model_dump() for row in request.telemetry])
    artifacts = ensure_artifacts()
    result = score_telemetry(frame, artifacts)
    return {
        "satellite_name": request.satellite_name,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        **result,
    }


@app.post("/simulate")
def simulate(request: SimulationRequest) -> dict:
    """Run digital twin action simulation from telemetry payload."""
    if not request.telemetry:
        return {"error": "telemetry payload must contain at least one sample"}

    frame = pd.DataFrame([row.model_dump() for row in request.telemetry])
    artifacts = ensure_artifacts()
    before = score_telemetry(frame, artifacts)
    twin_frame = simulate_action(frame, request.action, request.operator_mode)
    after = score_telemetry(twin_frame, artifacts)
    twin = compare_state(before["failure_risk_pct"], after["failure_risk_pct"])

    return {
        "satellite_name": request.satellite_name,
        "operator_mode": request.operator_mode,
        "action": request.action,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "before": before,
        "after": after,
        "digital_twin": twin,
    }
