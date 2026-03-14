"""Real-data telemetry feed with local caching and model-ready feature mapping."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.live_tracking import TrackPoint, get_track_point


def _load_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    except (json.JSONDecodeError, OSError):
        return []
    return []


def _save_cache(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _to_float(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _feature_map(point: TrackPoint, previous: dict | None) -> dict:
    """Map orbital state to normalized model telemetry features."""
    prev_alt = float(previous["altitude_km"]) if previous else point.altitude_km
    prev_vel = float(previous["velocity_kms"]) if previous else point.velocity_kms

    alt_delta = point.altitude_km - prev_alt
    vel_delta = point.velocity_kms - prev_vel
    latitude_load = abs(point.latitude) / 90.0

    # These mappings keep features in [0,1] similar to MinMax-scaled CMAPSS channels.
    temperature = _to_float(0.42 + latitude_load * 0.28 + abs(alt_delta) * 0.018, 0.0, 1.0)
    voltage = _to_float(0.86 - latitude_load * 0.22 - abs(vel_delta) * 0.50, 0.0, 1.0)
    vibration = _to_float(0.21 + abs(vel_delta) * 0.90 + abs(alt_delta) * 0.03, 0.0, 1.0)
    pressure = _norm(point.altitude_km, 380.0, 580.0)
    fuel_flow = _to_float(0.34 + latitude_load * 0.26 + abs(vel_delta) * 0.35, 0.0, 1.0)
    rpm = _to_float(0.70 - abs(vel_delta) * 0.55, 0.0, 1.0)

    health_index = 1.0 - (0.36 * temperature + 0.31 * vibration + 0.33 * (1.0 - voltage))
    rul_estimate = max(0.0, min(220.0, health_index * 220.0))

    return {
        "temperature": temperature,
        "voltage": voltage,
        "vibration": vibration,
        "pressure": pressure,
        "fuel_flow": fuel_flow,
        "rpm": rpm,
        "rul": rul_estimate,
    }


def append_live_sample(
    *,
    cache_path: Path,
    norad_id: int,
    unit_id: int,
    cycle_hint: float,
    timeout_sec: float,
    history_limit: int,
) -> tuple[pd.DataFrame, TrackPoint]:
    """Fetch one live orbital sample, append to cache, and return telemetry history."""
    records = _load_cache(cache_path)

    point = get_track_point(
        norad_id=norad_id,
        cycle=cycle_hint,
        unit_seed=unit_id,
        timeout_sec=timeout_sec,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    prev = records[-1] if records else None
    mapped = _feature_map(point, prev)
    next_cycle = (int(records[-1]["cycle"]) + 1) if records else 1

    record = {
        "unit_id": int(unit_id),
        "cycle": int(next_cycle),
        "timestamp_utc": point.timestamp_utc,
        "ingested_at": now_iso,
        "tracking_source": point.source,
        "is_live_tracking": bool(point.live),
        "latitude": float(point.latitude),
        "longitude": float(point.longitude),
        "altitude_km": float(point.altitude_km),
        "velocity_kms": float(point.velocity_kms),
        **mapped,
    }
    records.append(record)
    records = records[-int(history_limit) :]
    _save_cache(cache_path, records)

    return pd.DataFrame(records), point
