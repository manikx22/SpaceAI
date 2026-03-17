"""API tests for SpaceOps AI inference service."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["artifacts"]["lstm_model"] is True
    assert "ready" in payload


def test_readiness_and_manifest_endpoints() -> None:
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    ready_payload = ready.json()
    assert ready_payload["status"] in {"ready", "degraded"}
    assert "checks" in ready_payload

    manifest = client.get("/manifest")
    assert manifest.status_code == 200
    manifest_payload = manifest.json()
    assert manifest_payload["service"] == "SpaceOps AI"
    assert "artifacts" in manifest_payload


def test_predict_endpoint() -> None:
    payload = {
        "satellite_name": "SAT-TEST-01",
        "telemetry": [
            {
                "cycle": idx + 1,
                "temperature": 0.55,
                "voltage": 0.72,
                "vibration": 0.31,
                "pressure": 0.60,
                "fuel_flow": 0.45,
                "rpm": 0.58,
            }
            for idx in range(55)
        ],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["satellite_name"] == "SAT-TEST-01"
    assert body["health_state"] in {"HEALTHY", "WARNING", "CRITICAL"}
    assert "recommendation" in body
    assert "digital_twin" in body
    assert "explainability" in body
    assert "request_id" in body

    audit = client.get("/audit/recent")
    assert audit.status_code == 200
    audit_payload = audit.json()
    assert len(audit_payload["events"]) >= 1


def test_simulate_endpoint() -> None:
    payload = {
        "satellite_name": "SAT-TEST-01",
        "action": "Switch to power safe mode and rebalance battery bus.",
        "operator_mode": "Assist",
        "telemetry": [
            {
                "cycle": idx + 1,
                "temperature": 0.66,
                "voltage": 0.28,
                "vibration": 0.40,
                "pressure": 0.55,
                "fuel_flow": 0.52,
                "rpm": 0.54,
            }
            for idx in range(55)
        ],
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "before" in body
    assert "after" in body
    assert "digital_twin" in body
    assert "request_id" in body


def test_api_key_guard_when_enabled() -> None:
    previous_flag = config.REQUIRE_API_KEY
    previous_key = config.API_KEY
    config.REQUIRE_API_KEY = True
    config.API_KEY = "secret-token"
    try:
        unauthorized = client.get("/manifest")
        assert unauthorized.status_code == 401

        authorized = client.get("/manifest", headers={"X-API-Key": "secret-token"})
        assert authorized.status_code == 200
    finally:
        config.REQUIRE_API_KEY = previous_flag
        config.API_KEY = previous_key
