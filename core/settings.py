"""Environment-driven runtime settings for SpaceOps AI."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class RuntimeSettings:
    app_env: str
    log_level: str
    live_refresh_seconds: float
    tle_timeout_seconds: float
    tle_history_limit: int

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            app_env=os.getenv("SPACEOPS_APP_ENV", "dev"),
            log_level=os.getenv("SPACEOPS_LOG_LEVEL", "INFO").upper(),
            live_refresh_seconds=_env_float("SPACEOPS_LIVE_REFRESH_SECONDS", 0.11),
            tle_timeout_seconds=_env_float("SPACEOPS_TLE_TIMEOUT_SECONDS", 8.0),
            tle_history_limit=_env_int("SPACEOPS_TLE_HISTORY_LIMIT", 180),
        )
