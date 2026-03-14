"""Optional live space weather integration with resilient fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from urllib.error import URLError
from urllib.request import urlopen

import config


NOAA_KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"


def get_space_weather(timeout_sec: float = 4.0) -> dict:
    """Fetch live NOAA Kp index if reachable, else return cached/fallback data."""
    try:
        with urlopen(NOAA_KP_URL, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload:
            latest = payload[-1]
            result = {
                "source": "NOAA SWPC",
                "kp_index": float(latest.get("kp", 0.0)),
                "observed_at": latest.get("time_tag", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
                "live": True,
            }
            config.SPACE_WEATHER_CACHE_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
    except (URLError, TimeoutError, ValueError, OSError, KeyError):
        pass

    if config.SPACE_WEATHER_CACHE_PATH.exists():
        try:
            return json.loads(config.SPACE_WEATHER_CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "source": "fallback",
        "kp_index": 2.3,
        "observed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "live": False,
    }
