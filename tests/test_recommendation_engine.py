import pandas as pd

from recommendation_engine import generate_recommendation


def _history(temp=0.9, vib=0.9, volt=0.2):
    rows = []
    for i in range(30):
        rows.append(
            {
                "temperature": temp,
                "vibration": vib,
                "voltage": volt + (i * -0.001),
            }
        )
    return pd.DataFrame(rows)


def test_recommendation_returns_required_keys():
    history = _history()
    latest = history.iloc[-1]
    out = generate_recommendation(latest=latest, telemetry_history=history, anomaly_score=0.02, failure_risk_pct=40.0)

    assert "action" in out
    assert "confidence" in out
    assert "reason" in out


def test_recommendation_confidence_range():
    history = _history()
    latest = history.iloc[-1]
    out = generate_recommendation(latest=latest, telemetry_history=history, anomaly_score=0.06, failure_risk_pct=85.0)

    assert 0.0 <= out["confidence"] <= 1.0
