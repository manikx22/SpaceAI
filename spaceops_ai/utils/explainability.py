"""Explainability helpers for fault interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def classify_fault(latest: pd.Series, anomaly_score: float, failure_risk_pct: float) -> str:
    """Return a coarse fault class from model outputs and sensors."""
    if latest["voltage"] < config.VOLTAGE_LOW_THRESHOLD:
        return "Power Bus Instability"
    if latest["temperature"] > config.TEMP_THRESHOLD and latest["vibration"] > config.VIBRATION_THRESHOLD:
        return "Thermal Mechanical Stress"
    if failure_risk_pct > config.CRITICAL_FAILURE_RISK:
        return "High Failure Progression"
    if anomaly_score > config.WARNING_ANOMALY_SCORE:
        return "Telemetry Pattern Anomaly"
    return "Nominal Drift"


def severity_label(risk_pct: float, anomaly_score: float) -> str:
    """Map scores to severity label."""
    if risk_pct >= config.CRITICAL_FAILURE_RISK or anomaly_score >= config.CRITICAL_ANOMALY_SCORE:
        return "Critical"
    if risk_pct >= config.WARNING_FAILURE_RISK or anomaly_score >= config.WARNING_ANOMALY_SCORE:
        return "Warning"
    return "Low"


def explain_prediction(latest: pd.Series, baseline_df: pd.DataFrame) -> dict:
    """Return top drivers for the current prediction."""
    feature_scores: dict[str, float] = {}
    descriptions: dict[str, str] = {
        "temperature": "Heat level is above normal.",
        "voltage": "Power level is below normal.",
        "vibration": "Mechanical movement is rising.",
        "pressure": "Pressure trend is elevated.",
        "fuel_flow": "Subsystem demand is increasing.",
        "rpm": "Rotational performance is falling.",
    }

    for feature in config.TELEMETRY_FEATURES:
        mean = float(baseline_df[feature].mean())
        std = float(baseline_df[feature].std(ddof=0) or 1e-6)
        z = abs(float(latest[feature]) - mean) / std
        feature_scores[feature] = float(np.clip(z / 3.0, 0.0, 1.0))

    ranked = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)[: config.TOP_EXPLANATION_FACTORS]
    top_factors = [
        {
            "feature": feature,
            "score": round(score, 4),
            "message": descriptions[feature],
            "value": round(float(latest[feature]), 4),
        }
        for feature, score in ranked
    ]
    summary = " | ".join(f"{item['feature']}: {item['message']}" for item in top_factors)

    return {"top_factors": top_factors, "feature_scores": feature_scores, "summary": summary}
