"""Shared inference utilities for the dashboard and API."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

import config
import preprocess
from recommendation_engine import generate_recommendation
from train_lstm import LSTMRegressor
from utils.digital_twin import compare_state, simulate_action
from utils.explainability import classify_fault, explain_prediction, severity_label


class Autoencoder(nn.Module):
    """Inference mirror of the trained anomaly autoencoder."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class ArtifactBundle:
    """Loaded model artifacts used for prediction."""

    df: pd.DataFrame
    anomaly_bundle: dict | None
    iforest: object | None
    lstm_bundle: dict | None


def ensure_artifacts() -> ArtifactBundle:
    """Load model artifacts and bootstrap processed data if required."""
    if not config.PROCESSED_CSV_PATH.exists():
        preprocess.main()

    df = pd.read_csv(config.PROCESSED_CSV_PATH)
    anomaly_bundle = torch.load(config.ANOMALY_MODEL_PATH, map_location="cpu") if config.ANOMALY_MODEL_PATH.exists() else None
    iforest = joblib.load(config.ANOMALY_IFOREST_PATH) if config.ANOMALY_IFOREST_PATH.exists() else None
    lstm_bundle = torch.load(config.LSTM_MODEL_PATH, map_location="cpu") if config.LSTM_MODEL_PATH.exists() else None
    return ArtifactBundle(df=df, anomaly_bundle=anomaly_bundle, iforest=iforest, lstm_bundle=lstm_bundle)


def add_extended_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived mission channels used by the UI and API outputs."""
    ext = df.copy()
    ext["radiation"] = np.clip(0.30 + 0.54 * ext["temperature"] + 0.08 * np.sin(ext["cycle"] / 13), 0, 1)
    ext["battery"] = np.clip(0.78 * ext["voltage"] + 0.16 * np.cos(ext["cycle"] / 22), 0, 1)
    ext["solar_efficiency"] = np.clip(0.42 + 0.53 * ext["battery"] - 0.14 * ext["vibration"], 0, 1)
    return ext


def compute_anomaly_score(feature_vec: np.ndarray, anomaly_bundle, iforest) -> float:
    """Compute anomaly score using the best available anomaly model."""
    if anomaly_bundle is not None:
        model = Autoencoder(anomaly_bundle["input_dim"])
        model.load_state_dict(anomaly_bundle["model_state_dict"])
        model.eval()
        x = torch.tensor(feature_vec[None, :], dtype=torch.float32)
        with torch.no_grad():
            recon = model(x)
        return float(torch.mean((x - recon) ** 2).item())

    if iforest is not None:
        score = float(iforest.decision_function(feature_vec[None, :])[0])
        return max(0.0, -score)

    return 0.0


def predict_rul(window: np.ndarray, lstm_bundle) -> float:
    """Predict remaining useful life from a telemetry window."""
    if lstm_bundle is None:
        return float("nan")

    model = LSTMRegressor(
        input_size=lstm_bundle["input_size"],
        hidden_size=lstm_bundle["hidden_size"],
        num_layers=lstm_bundle["num_layers"],
    )
    model.load_state_dict(lstm_bundle["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(window[None, :, :], dtype=torch.float32)).item()
    return float(max(0.0, pred))


def classify_health(anomaly_score: float, failure_risk_pct: float) -> str:
    """Convert AI outputs into a health label."""
    if anomaly_score >= config.CRITICAL_ANOMALY_SCORE or failure_risk_pct >= config.CRITICAL_FAILURE_RISK:
        return "CRITICAL"
    if anomaly_score >= config.WARNING_ANOMALY_SCORE or failure_risk_pct >= config.WARNING_FAILURE_RISK:
        return "WARNING"
    return "HEALTHY"


def _prepare_window(df: pd.DataFrame) -> np.ndarray:
    """Prepare a fixed-size model input window."""
    window = df[config.TELEMETRY_FEATURES].tail(config.WINDOW_SIZE).to_numpy(dtype=np.float32)
    if len(window) < config.WINDOW_SIZE:
        pad = np.repeat(window[:1], config.WINDOW_SIZE - len(window), axis=0)
        window = np.vstack([pad, window])
    return window


def score_telemetry(history_df: pd.DataFrame, artifacts: ArtifactBundle) -> dict:
    """Run full prediction stack for a telemetry history."""
    frame = history_df.copy()
    if "cycle" not in frame.columns:
        frame.insert(0, "cycle", np.arange(1, len(frame) + 1))
    frame = add_extended_channels(frame)

    latest = frame.iloc[-1]
    feature_vec = latest[config.TELEMETRY_FEATURES].to_numpy(dtype=np.float32)
    anomaly_score = compute_anomaly_score(feature_vec, artifacts.anomaly_bundle, artifacts.iforest)
    predicted_rul = predict_rul(_prepare_window(frame), artifacts.lstm_bundle)

    baseline_df = artifacts.df
    max_rul = max(1.0, float(baseline_df["rul"].max())) if "rul" in baseline_df.columns else 1.0
    failure_risk = float(np.clip(100.0 * (1.0 - predicted_rul / max_rul), 0.0, 100.0)) if not np.isnan(predicted_rul) else 0.0
    health = classify_health(anomaly_score, failure_risk)
    recommendation = generate_recommendation(latest=latest, telemetry_history=frame, anomaly_score=anomaly_score, failure_risk_pct=failure_risk)
    fault_class = classify_fault(latest, anomaly_score, failure_risk)
    severity = severity_label(failure_risk, anomaly_score)
    explainability = explain_prediction(latest, artifacts.df)
    twin_frame = simulate_action(frame, recommendation["action"], "Assist")
    twin_latest = twin_frame.iloc[-1]
    twin_feature_vec = twin_latest[config.TELEMETRY_FEATURES].to_numpy(dtype=np.float32)
    twin_anomaly = compute_anomaly_score(twin_feature_vec, artifacts.anomaly_bundle, artifacts.iforest)
    twin_rul = predict_rul(_prepare_window(twin_frame), artifacts.lstm_bundle)
    twin_risk = float(np.clip(100.0 * (1.0 - twin_rul / max_rul), 0.0, 100.0)) if not np.isnan(twin_rul) else 0.0
    digital_twin = compare_state(failure_risk, twin_risk)

    return {
        "health_state": health,
        "fault_class": fault_class,
        "severity": severity,
        "anomaly_score": float(anomaly_score),
        "predicted_rul": float(predicted_rul),
        "failure_risk_pct": float(failure_risk),
        "recommendation": recommendation,
        "explainability": explainability,
        "digital_twin": digital_twin,
        "latest_snapshot": latest.to_dict(),
    }
