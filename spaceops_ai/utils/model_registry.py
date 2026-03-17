"""Model manifest and readiness helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    modified = None
    size_bytes = 0
    if exists:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        size_bytes = int(stat.st_size)
    return {
        "path": str(path),
        "exists": exists,
        "modified_utc": modified,
        "size_bytes": size_bytes,
    }


def build_manifest() -> dict[str, Any]:
    """Build a deployment-friendly manifest for models and processed artifacts."""
    anomaly_report = _load_json(config.ANOMALY_REPORT_PATH)
    lstm_report = _load_json(config.LSTM_REPORT_PATH)
    quality_report = _load_json(config.DATA_QUALITY_REPORT_PATH)

    artifacts = {
        "processed_dataset": _artifact_status(config.PROCESSED_CSV_PATH),
        "scaler": _artifact_status(config.SCALER_PATH),
        "anomaly_model": _artifact_status(config.ANOMALY_MODEL_PATH),
        "iforest_model": _artifact_status(config.ANOMALY_IFOREST_PATH),
        "lstm_model": _artifact_status(config.LSTM_MODEL_PATH),
    }
    ready = artifacts["processed_dataset"]["exists"] and (
        artifacts["anomaly_model"]["exists"] or artifacts["iforest_model"]["exists"]
    ) and artifacts["lstm_model"]["exists"]

    return {
        "service": "SpaceOps AI",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "environment": config.DEPLOYMENT_ENV,
        "ready": ready,
        "window_size": config.WINDOW_SIZE,
        "features": config.TELEMETRY_FEATURES,
        "artifacts": artifacts,
        "reports": {
            "data_quality": quality_report,
            "anomaly_training": anomaly_report,
            "lstm_training": lstm_report,
        },
    }


def persist_manifest(path: Path | None = None) -> dict[str, Any]:
    """Persist the current model manifest for external inspection."""
    target = path or config.MODEL_MANIFEST_PATH
    manifest = build_manifest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
