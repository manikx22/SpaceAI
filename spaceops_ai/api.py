"""FastAPI service for SpaceOps AI inference and project health."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

import config
from utils.digital_twin import compare_state, simulate_action
from utils.audit import append_audit_event, load_recent_events
from utils.inference import ensure_artifacts, score_telemetry
from utils.model_registry import persist_manifest


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
    telemetry: list[TelemetryPoint] = Field(max_length=config.MAX_TELEMETRY_BATCH)


class SimulationRequest(PredictionRequest):
    """Payload for action simulation."""

    action: str
    operator_mode: str = Field(default="Assist")


app = FastAPI(title="SpaceOps AI API", version="1.0.0")


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Attach a stable request identifier to every response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def enforce_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require an API key when the service is configured for authenticated mode."""
    if not config.REQUIRE_API_KEY:
        return
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing api key")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _audit(request: Request, endpoint: str, payload: dict) -> None:
    append_audit_event(
        config.API_AUDIT_LOG_PATH,
        {
            "request_id": getattr(request.state, "request_id", "unknown"),
            "endpoint": endpoint,
            **payload,
        },
    )


@app.get("/health")
def health() -> dict:
    """Health endpoint for deployment/runtime checks."""
    artifacts = ensure_artifacts()
    manifest = persist_manifest()
    return {
        "status": "ok",
        "timestamp_utc": _timestamp(),
        "processed_rows": int(len(artifacts.df)),
        "artifacts": {
            "anomaly_model": config.ANOMALY_MODEL_PATH.exists(),
            "iforest_model": config.ANOMALY_IFOREST_PATH.exists(),
            "lstm_model": config.LSTM_MODEL_PATH.exists(),
        },
        "ready": manifest["ready"],
        "environment": config.DEPLOYMENT_ENV,
    }


@app.get("/health/live")
def liveness() -> dict:
    """Liveness probe for container/runtime checks."""
    return {"status": "alive", "timestamp_utc": _timestamp()}


@app.get("/health/ready")
def readiness() -> dict:
    """Readiness probe exposing artifact state."""
    manifest = persist_manifest()
    return {
        "status": "ready" if manifest["ready"] else "degraded",
        "timestamp_utc": _timestamp(),
        "manifest_path": str(config.MODEL_MANIFEST_PATH),
        "checks": manifest["artifacts"],
    }


@app.get("/manifest")
def manifest(_: None = Depends(enforce_api_key)) -> dict:
    """Return the current model manifest."""
    return persist_manifest()


@app.get("/audit/recent")
def recent_audit(_: None = Depends(enforce_api_key)) -> dict:
    """Expose recent audit events for debugging and traceability."""
    return {"events": load_recent_events(config.API_AUDIT_LOG_PATH, limit=25)}


@app.post("/predict")
def predict(request: PredictionRequest, http_request: Request, _: None = Depends(enforce_api_key)) -> dict:
    """Run anomaly detection, failure prediction, and recommendation inference."""
    if not request.telemetry:
        return {"error": "telemetry payload must contain at least one sample"}

    frame = pd.DataFrame([row.model_dump() for row in request.telemetry])
    artifacts = ensure_artifacts()
    result = score_telemetry(frame, artifacts)
    response = {
        "satellite_name": request.satellite_name,
        "timestamp_utc": _timestamp(),
        "request_id": http_request.state.request_id,
        **result,
    }
    _audit(
        http_request,
        "/predict",
        {
            "satellite_name": request.satellite_name,
            "samples": len(request.telemetry),
            "health_state": result["health_state"],
            "failure_risk_pct": result["failure_risk_pct"],
            "fault_class": result["fault_class"],
        },
    )
    return response


@app.post("/simulate")
def simulate(request: SimulationRequest, http_request: Request, _: None = Depends(enforce_api_key)) -> dict:
    """Run digital twin action simulation from telemetry payload."""
    if not request.telemetry:
        return {"error": "telemetry payload must contain at least one sample"}

    frame = pd.DataFrame([row.model_dump() for row in request.telemetry])
    artifacts = ensure_artifacts()
    before = score_telemetry(frame, artifacts)
    twin_frame = simulate_action(frame, request.action, request.operator_mode)
    after = score_telemetry(twin_frame, artifacts)
    twin = compare_state(before["failure_risk_pct"], after["failure_risk_pct"])

    response = {
        "satellite_name": request.satellite_name,
        "operator_mode": request.operator_mode,
        "action": request.action,
        "timestamp_utc": _timestamp(),
        "request_id": http_request.state.request_id,
        "before": before,
        "after": after,
        "digital_twin": twin,
    }
    _audit(
        http_request,
        "/simulate",
        {
            "satellite_name": request.satellite_name,
            "samples": len(request.telemetry),
            "operator_mode": request.operator_mode,
            "action": request.action,
            "risk_before": before["failure_risk_pct"],
            "risk_after": after["failure_risk_pct"],
            "risk_delta": twin["risk_reduction"],
        },
    )
    return response
