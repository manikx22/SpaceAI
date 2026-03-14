"""Mission timeline builder for incident replay."""

from __future__ import annotations


def build_timeline(*, health: str, risk: float, drift_status: str, action: str, operator_mode: str, live: bool) -> list[dict]:
    """Create a simple event timeline for the current mission state."""
    timeline = [
        {"stage": "Telemetry Ingested", "detail": "Sensor stream synchronized.", "status": "done"},
        {"stage": "AI Scan", "detail": f"Health={health}, Risk={risk:.1f}%, Drift={drift_status}.", "status": "done"},
    ]

    if health != "HEALTHY" or drift_status != "NOMINAL":
        timeline.append({"stage": "Fault Flag", "detail": "System raised an active operational alert.", "status": "done"})
    else:
        timeline.append({"stage": "Fault Flag", "detail": "No urgent alert raised.", "status": "standby"})

    timeline.append(
        {
            "stage": "Action Planned",
            "detail": f"{action} | Mode={operator_mode} | Link={'Live' if live else 'Simulated'}",
            "status": "done",
        }
    )
    timeline.append({"stage": "Impact Review", "detail": "Digital twin estimates mission outcome after action.", "status": "done"})
    return timeline
