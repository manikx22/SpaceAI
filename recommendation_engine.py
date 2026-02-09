"""Rule-based recommendation engine for self-healing actions."""

from __future__ import annotations

import pandas as pd

import config


def _voltage_trend_is_dropping(history: pd.DataFrame, lookback: int = 20) -> bool:
    """Detect persistent voltage drop using average first difference."""
    if len(history) < max(3, lookback):
        return False
    tail = history["voltage"].tail(lookback)
    slope_proxy = tail.diff().mean()
    return float(slope_proxy) < config.VOLTAGE_DROP_TREND


def generate_recommendation(
    latest: pd.Series,
    telemetry_history: pd.DataFrame,
    anomaly_score: float,
    failure_risk_pct: float,
) -> dict:
    """Return best recommendation and confidence from predefined rules."""
    rules = []

    if latest["temperature"] > config.TEMP_THRESHOLD and latest["vibration"] > config.VIBRATION_THRESHOLD:
        rules.append(
            {
                "action": "Reduce payload load and reduce attitude maneuver frequency.",
                "confidence": 0.88,
                "reason": "High thermal and vibration stress indicates mechanical overloading.",
            }
        )

    if _voltage_trend_is_dropping(telemetry_history) or latest["voltage"] < config.VOLTAGE_LOW_THRESHOLD:
        rules.append(
            {
                "action": "Switch to power safe mode and rebalance battery bus.",
                "confidence": 0.82,
                "reason": "Voltage decay trend suggests potential power subsystem instability.",
            }
        )

    if anomaly_score > config.CRITICAL_ANOMALY_SCORE or failure_risk_pct > config.CRITICAL_FAILURE_RISK:
        rules.append(
            {
                "action": "Enter autonomous contingency mode and schedule immediate diagnostic downlink.",
                "confidence": 0.91,
                "reason": "Critical AI risk signals require aggressive fault containment.",
            }
        )

    if not rules:
        rules.append(
            {
                "action": "Continue nominal operations with increased telemetry sampling.",
                "confidence": 0.64,
                "reason": "No critical rule triggered, monitoring recommended.",
            }
        )

    best = max(rules, key=lambda r: r["confidence"])
    return best
