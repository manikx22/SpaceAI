"""Live satellite tracking helpers with resilient network fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np


ISS_NORAD_ID = 25544
ISS_API_URL = "https://api.wheretheiss.at/v1/satellites/25544"


@dataclass
class TrackPoint:
    """Single satellite position sample for UI rendering."""

    latitude: float
    longitude: float
    altitude_km: float
    velocity_kms: float
    timestamp_utc: str
    source: str
    live: bool
    message: str


def simulated_track(cycle: float, unit_seed: int) -> TrackPoint:
    """Generate deterministic fallback orbital state."""
    phase = 0.16 * unit_seed
    latitude = float(52.0 * np.sin(0.11 * cycle + phase))
    longitude = float(((cycle * 3.4 + unit_seed * 13.0) % 360.0) - 180.0)
    altitude = float(540.0 + 20.0 * np.sin(cycle / 28.0))
    velocity = float(7.60 + 0.14 * np.cos(cycle / 17.0))

    return TrackPoint(
        latitude=latitude,
        longitude=longitude,
        altitude_km=altitude,
        velocity_kms=velocity,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        source="simulation",
        live=False,
        message="Simulated track active.",
    )


def fetch_live_iss(timeout_sec: float = 6.0) -> TrackPoint:
    """Fetch real-time ISS position from public API."""
    with urlopen(ISS_API_URL, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))

    timestamp = datetime.fromtimestamp(int(payload["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return TrackPoint(
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        altitude_km=float(payload.get("altitude", 420.0)),
        velocity_kms=float(payload.get("velocity", 27600.0)) / 3600.0,
        timestamp_utc=timestamp,
        source="wheretheiss.at",
        live=True,
        message="Live orbital feed active.",
    )


def get_track_point(norad_id: int, cycle: float, unit_seed: int, timeout_sec: float = 6.0) -> TrackPoint:
    """Return live track if available for supported satellites, else fallback."""
    if int(norad_id) != ISS_NORAD_ID:
        point = simulated_track(cycle, unit_seed)
        point.message = "Live feed unavailable for this satellite in demo mode. Using simulated orbit."
        return point

    try:
        return fetch_live_iss(timeout_sec=timeout_sec)
    except (URLError, TimeoutError, ValueError, KeyError):
        point = simulated_track(cycle, unit_seed)
        point.message = "Could not reach live ISS feed. Using simulated orbit."
        return point
