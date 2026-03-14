"""Persistent alert history utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def load_alert_history(path: Path) -> list[dict]:
    """Read alert history from disk."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    except (json.JSONDecodeError, OSError):
        return []
    return []


def append_alert(path: Path, alert: dict, limit: int) -> list[dict]:
    """Append alert with dedup and keep bounded history."""
    history = load_alert_history(path)
    enriched = {"timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), **alert}

    if history:
        prev = history[-1]
        # Deduplicate identical consecutive alert states.
        if all(prev.get(k) == enriched.get(k) for k in ["type", "severity", "satellite", "message"]):
            return history

    history.append(enriched)
    history = history[-int(limit) :]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
