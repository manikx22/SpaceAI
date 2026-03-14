"""Digital twin action simulation for recommendation outcome testing."""

from __future__ import annotations

import numpy as np
import pandas as pd


ACTION_LIBRARY = {
    "Reduce payload load and reduce attitude maneuver frequency.": {
        "temperature": -0.10,
        "vibration": -0.12,
        "voltage": 0.04,
    },
    "Switch to power safe mode and rebalance battery bus.": {
        "voltage": 0.12,
        "fuel_flow": -0.06,
        "rpm": -0.04,
    },
    "Enter autonomous contingency mode and schedule immediate diagnostic downlink.": {
        "temperature": -0.08,
        "vibration": -0.08,
        "voltage": 0.08,
        "pressure": -0.04,
    },
    "Continue nominal operations with increased telemetry sampling.": {
        "temperature": -0.01,
        "vibration": -0.01,
    },
}


MODE_EFFECT = {
    "Manual": 0.85,
    "Assist": 1.0,
    "Autonomous": 1.15,
}


def simulate_action(frame: pd.DataFrame, action: str, mode: str) -> pd.DataFrame:
    """Apply recommended action to the latest telemetry window."""
    result = frame.copy()
    if result.empty:
        return result

    latest_idx = result.index[-1]
    effects = ACTION_LIBRARY.get(action, {})
    factor = MODE_EFFECT.get(mode, 1.0)

    for feature, delta in effects.items():
        result.loc[latest_idx, feature] = np.clip(float(result.loc[latest_idx, feature]) + delta * factor, 0.0, 1.0)

    return result


def compare_state(before_risk: float, after_risk: float) -> dict:
    """Return digital twin delta metrics."""
    change = float(before_risk - after_risk)
    return {
        "risk_before": round(before_risk, 3),
        "risk_after": round(after_risk, 3),
        "risk_reduction": round(change, 3),
        "improved": change > 0.0,
    }
