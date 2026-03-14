"""Smoke tests for SpaceOps AI core flows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import preprocess
from utils.alerts import append_alert, load_alert_history
from utils.explainability import classify_fault, explain_prediction, severity_label
from utils.model_monitoring import baseline_stats, compute_drift_report
from utils.real_telemetry import append_live_sample
from utils.scenario_engine import apply_scenario
from utils.space_weather import get_space_weather


def test_preprocess_outputs_exist() -> None:
    preprocess.main()
    assert config.PROCESSED_CSV_PATH.exists()
    assert config.SCALER_PATH.exists()
    assert config.DATA_QUALITY_REPORT_PATH.exists()

    payload = json.loads(config.DATA_QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
    assert "quality_score" in payload
    assert payload["quality_score"] >= 0.0


def test_live_telemetry_append_schema() -> None:
    df, point = append_live_sample(
        cache_path=config.LIVE_TELEMETRY_CACHE_PATH,
        norad_id=20580,
        unit_id=1,
        cycle_hint=1.0,
        timeout_sec=2.0,
        history_limit=25,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
    for col in ["temperature", "voltage", "vibration", "pressure", "fuel_flow", "rpm", "rul", "latitude", "longitude"]:
        assert col in df.columns
    assert point.source in {"simulation", "wheretheiss.at"}


def test_drift_report_generation() -> None:
    df = pd.read_csv(config.PROCESSED_CSV_PATH)
    means, stds = baseline_stats(df, config.TELEMETRY_FEATURES)
    current = df.tail(60).copy()

    report = compute_drift_report(
        baseline_means=means,
        baseline_stds=stds,
        current_df=current,
        features=config.TELEMETRY_FEATURES,
        warn_threshold=config.DRIFT_WARNING_THRESHOLD,
        critical_threshold=config.DRIFT_CRITICAL_THRESHOLD,
    )
    assert 0.0 <= report.score <= 1.0
    assert report.status in {"NOMINAL", "WARNING", "CRITICAL"}
    assert set(report.feature_scores.keys()) == set(config.TELEMETRY_FEATURES)


def test_alert_history_append() -> None:
    history = append_alert(
        config.ALERT_HISTORY_PATH,
        {"type": "ops_alert", "severity": "WARNING", "satellite": "SAT-TEST", "message": "Test alert"},
        config.ALERT_HISTORY_LIMIT,
    )
    loaded = load_alert_history(config.ALERT_HISTORY_PATH)
    assert len(history) >= 1
    assert len(loaded) >= 1
    assert loaded[-1]["severity"] in {"INFO", "WARNING", "CRITICAL"}


def test_scenario_and_explainability() -> None:
    df = pd.read_csv(config.PROCESSED_CSV_PATH).tail(60).reset_index(drop=True)
    stressed = apply_scenario(df, "Thermal Spike")
    latest = stressed.iloc[-1]
    explanation = explain_prediction(latest, df)
    fault = classify_fault(latest, anomaly_score=0.03, failure_risk_pct=61.0)
    severity = severity_label(61.0, 0.03)

    assert len(explanation["top_factors"]) >= 1
    assert isinstance(fault, str)
    assert severity in {"Low", "Warning", "Critical"}


def test_space_weather_fallback_shape() -> None:
    weather = get_space_weather(timeout_sec=0.001)
    assert "kp_index" in weather
    assert "source" in weather
