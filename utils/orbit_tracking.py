"""Live satellite orbit tracking via CelesTrak TLE + Skyfield.

This module is optional at runtime: it gracefully degrades when network
or dependencies are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests


TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"


@dataclass
class OrbitState:
    source: str
    norad_id: int
    timestamp_utc: str
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    speed_kms: float


def _fetch_tle_lines(norad_id: int, timeout_sec: float = 8.0) -> tuple[str, str]:
    """Fetch 2-line element set from CelesTrak for a NORAD id."""
    url = TLE_URL.format(norad_id=norad_id)
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()

    lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
    tle_lines = [line for line in lines if line.startswith("1 ") or line.startswith("2 ")]

    if len(tle_lines) < 2:
        raise ValueError(f"No valid TLE returned for NORAD {norad_id}")

    return tle_lines[0], tle_lines[1]


def get_live_orbit_state(norad_id: int = 25544) -> Optional[OrbitState]:
    """Compute real-time subpoint and speed from latest CelesTrak TLE.

    Returns None on any dependency/network/data error.
    """
    try:
        from skyfield.api import EarthSatellite, load
    except Exception:
        return None

    try:
        l1, l2 = _fetch_tle_lines(norad_id)
        ts = load.timescale()
        t = ts.now()

        sat = EarthSatellite(l1, l2, f"NORAD-{norad_id}", ts)
        geo = sat.at(t)

        subpoint = geo.subpoint()
        lat = float(subpoint.latitude.degrees)
        lon = float(subpoint.longitude.degrees)
        alt_km = float(subpoint.elevation.km)

        # Speed from inertial velocity vector magnitude.
        speed_kms = float((geo.velocity.km_per_s**2).sum() ** 0.5)

        return OrbitState(
            source="CelesTrak+Skyfield",
            norad_id=norad_id,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_km=alt_km,
            speed_kms=speed_kms,
        )
    except Exception:
        return None
