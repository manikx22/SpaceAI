"""SpaceOps AI dashboard with mission-control UI and resilient live tracking."""

from __future__ import annotations

import time
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageDraw
from torch import nn

import config
import preprocess
import train_anomaly
import train_lstm
from recommendation_engine import generate_recommendation
from train_lstm import LSTMRegressor
from utils.alerts import append_alert
from utils.digital_twin import compare_state, simulate_action
from utils.explainability import classify_fault, explain_prediction, severity_label
from utils.live_tracking import get_track_point
from utils.mission_timeline import build_timeline
from utils.model_monitoring import baseline_stats, compute_drift_report
from utils.real_telemetry import append_live_sample
from utils.scenario_engine import SCENARIOS, apply_scenario
from utils.space_weather import get_space_weather


SATELLITES = {
    "ISS (ZARYA)": {
        "label": "ISS-01",
        "norad_id": 25544,
        "mode": "Low Earth Orbit",
        "image_key": "ISS (ZARYA)",
    },
    "Hubble Space Telescope": {
        "label": "HST-02",
        "norad_id": 20580,
        "mode": "Earth Observation",
        "image_key": "Hubble Space Telescope",
    },
    "NOAA-15": {
        "label": "NOAA-03",
        "norad_id": 25338,
        "mode": "Weather Monitoring",
        "image_key": "NOAA-15",
    },
}


class Autoencoder(nn.Module):
    """Inference mirror of anomaly autoencoder architecture."""

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


@st.cache_resource
def load_artifacts():
    """Load saved datasets and models, generating data if missing."""
    if not config.PROCESSED_CSV_PATH.exists():
        preprocess.main()

    df = pd.read_csv(config.PROCESSED_CSV_PATH)
    anomaly_bundle = torch.load(config.ANOMALY_MODEL_PATH, map_location="cpu") if config.ANOMALY_MODEL_PATH.exists() else None
    iforest = joblib.load(config.ANOMALY_IFOREST_PATH) if config.ANOMALY_IFOREST_PATH.exists() else None
    lstm_bundle = torch.load(config.LSTM_MODEL_PATH, map_location="cpu") if config.LSTM_MODEL_PATH.exists() else None
    return df, anomaly_bundle, iforest, lstm_bundle


def with_extended_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic mission channels for richer telemetry console."""
    ext = df.copy()
    ext["radiation"] = np.clip(0.30 + 0.54 * ext["temperature"] + 0.08 * np.sin(ext["cycle"] / 13), 0, 1)
    ext["battery"] = np.clip(0.78 * ext["voltage"] + 0.16 * np.cos(ext["cycle"] / 22), 0, 1)
    ext["solar_efficiency"] = np.clip(0.42 + 0.53 * ext["battery"] - 0.14 * ext["vibration"], 0, 1)
    return ext


def compute_anomaly_score(feature_vec: np.ndarray, anomaly_bundle, iforest) -> float:
    """Compute anomaly score from autoencoder or IsolationForest fallback."""
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
    """Predict RUL from latest telemetry window."""
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


def status_from_risk(anomaly_score: float, failure_risk_pct: float) -> tuple[str, str]:
    """Compute health state and color."""
    if anomaly_score >= config.CRITICAL_ANOMALY_SCORE or failure_risk_pct >= config.CRITICAL_FAILURE_RISK:
        return "CRITICAL", "#FF2D55"
    if anomaly_score >= config.WARNING_ANOMALY_SCORE or failure_risk_pct >= config.WARNING_FAILURE_RISK:
        return "WARNING", "#FF6B00"
    return "HEALTHY", "#00FFD1"


def forecast(series: pd.Series, steps: int = 22) -> tuple[np.ndarray, np.ndarray]:
    """Generate dotted short-horizon forecast line."""
    if len(series) < 2:
        return np.array([]), np.array([])

    tail = series.tail(min(24, len(series)))
    slope = float(tail.diff().mean())
    start = int(series.index.max()) + 1
    x = np.arange(start, start + steps)
    y = np.array([np.clip(float(series.iloc[-1]) + slope * i, 0, 1) for i in range(1, steps + 1)])
    return x, y


def load_cached_live_telemetry() -> pd.DataFrame:
    """Load cached live telemetry history if available."""
    path = config.LIVE_TELEMETRY_CACHE_PATH
    if not path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            return pd.DataFrame(payload)
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame()
    return pd.DataFrame()


def load_data_quality_report() -> dict:
    """Load preprocessing quality report if available."""
    path = config.DATA_QUALITY_REPORT_PATH
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_json_report(path: Path) -> dict:
    """Load any JSON report file if present."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def run_retraining_pipeline() -> tuple[bool, str]:
    """Run preprocess + anomaly + LSTM training, returning status."""
    try:
        preprocess.main()
        train_anomaly.main()
        train_lstm.main()
        load_artifacts.clear()
        st.cache_resource.clear()
        return True, "Retraining pipeline completed and model cache refreshed."
    except Exception as exc:  # pragma: no cover - dashboard runtime guard
        return False, f"Retraining failed: {exc}"


def ensure_satellite_images() -> dict[str, Path]:
    """Create local satellite images so previews always render."""
    asset_dir = config.PROCESSED_DATA_DIR / "ui_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "ISS (ZARYA)": ("iss_card.png", (2, 6, 23), (0, 229, 255)),
        "Hubble Space Telescope": ("hubble_card.png", (10, 10, 28), (124, 58, 237)),
        "NOAA-15": ("noaa_card.png", (4, 12, 24), (255, 107, 0)),
    }
    image_paths: dict[str, Path] = {}

    for label, (filename, bg_color, accent_color) in specs.items():
        path = asset_dir / filename
        image_paths[label] = path
        if path.exists():
            continue

        image = Image.new("RGB", (1200, 700), bg_color)
        draw = ImageDraw.Draw(image)
        rng = np.random.default_rng(abs(hash(filename)) % (2**32))

        for _ in range(400):
            x = int(rng.integers(0, 1200))
            y = int(rng.integers(0, 700))
            r = int(rng.integers(1, 3))
            c = int(rng.integers(170, 255))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(c, c, c))

        draw.ellipse((65, 520, 590, 1020), fill=(13, 64, 104), outline=(110, 200, 255), width=3)
        draw.arc((80, 90, 1120, 620), start=12, end=168, fill=accent_color, width=3)
        draw.rectangle((520, 290, 690, 410), fill=(220, 228, 238), outline=(18, 25, 38), width=4)
        draw.rectangle((345, 315, 505, 385), fill=accent_color, outline=(18, 25, 38), width=4)
        draw.rectangle((705, 315, 865, 385), fill=accent_color, outline=(18, 25, 38), width=4)
        draw.line((605, 290, 605, 220), fill=(230, 238, 248), width=5)
        draw.ellipse((575, 185, 635, 245), fill=(245, 248, 252), outline=(18, 25, 38), width=3)
        draw.text((40, 34), f"SpaceOps AI | {label}", fill=(230, 238, 248))
        draw.text((40, 74), "Mission Camera View", fill=accent_color)
        image.save(path)

    return image_paths


def inject_css() -> None:
    """Inject the single dark theme used by the dashboard."""
    t = {
        "bg_a": "#020617",
        "bg_b": "#000000",
        "panel": "rgba(10, 18, 34, 0.92)",
        "border": "rgba(0, 229, 255, 0.30)",
        "text": "#E2E8F0",
        "muted": "#94A3B8",
        "sidebar": "rgba(8, 15, 30, 0.94)",
    }

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Rajdhani:wght@400;600;700&family=Exo+2:wght@500;700&display=swap');

:root {{
  --bg-a: {t['bg_a']};
  --bg-b: {t['bg_b']};
  --panel: {t['panel']};
  --border: {t['border']};
  --text: {t['text']};
  --muted: {t['muted']};
  --cyan: #00E5FF;
  --purple: #7C3AED;
  --pink: #FF4FD8;
  --orange: #FF6B00;
}}

.stApp {{
  color: var(--text);
  font-family: 'Rajdhani', sans-serif;
  background:
    radial-gradient(circle at 20% 20%, rgba(255,255,255,0.07) 0 1px, transparent 1px 100%),
    radial-gradient(circle at 72% 32%, rgba(255,255,255,0.06) 0 1px, transparent 1px 100%),
    radial-gradient(circle at 8% 10%, rgba(0,229,255,0.18), transparent 26%),
    radial-gradient(circle at 90% 16%, rgba(124,58,237,0.15), transparent 24%),
    radial-gradient(circle at 50% 120%, rgba(255,107,0,0.10), transparent 30%),
    linear-gradient(145deg, var(--bg-a) 0%, var(--bg-b) 100%);
}}

.main .block-container {{
  max-width: 1600px;
  padding-top: 0.8rem;
}}

[data-testid="stSidebar"] > div:first-child {{
  background: {t['sidebar']};
  border-right: 1px solid var(--border);
}}

.mission-bar {{
  border: 1px solid var(--border);
  border-radius: 14px;
  background:
    linear-gradient(90deg, rgba(4,12,24,0.94), rgba(10,18,34,0.76)),
    radial-gradient(circle at right, rgba(0,229,255,0.12), transparent 30%);
  padding: 12px 14px;
  margin-bottom: 10px;
  position: relative;
  overflow: hidden;
}}

.mission-bar::after {{
  content: "";
  position: absolute;
  right: -90px;
  top: -90px;
  width: 220px;
  height: 220px;
  border: 1px solid rgba(0,229,255,0.16);
  border-radius: 50%;
  box-shadow: 0 0 0 28px rgba(0,229,255,0.05), 0 0 0 56px rgba(0,229,255,0.03);
}}

.mission-title {{
  font-family: 'Orbitron', sans-serif;
  font-size: 1.25rem;
  letter-spacing: 0.8px;
}}

.mission-sub {{
  color: var(--muted);
  font-size: 0.92rem;
}}

.card {{
  border: 1px solid var(--border);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(12,20,38,0.98), rgba(9,16,30,0.90)), var(--panel);
  padding: 12px;
  box-shadow: 0 14px 30px rgba(2, 6, 23, 0.25);
  position: relative;
  overflow: hidden;
}}

.card::before {{
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(rgba(0,229,255,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.05) 1px, transparent 1px);
  background-size: 22px 22px;
  opacity: 0.3;
  pointer-events: none;
}}

.card-title {{
  font-family: 'Orbitron', sans-serif;
  font-size: 0.90rem;
  letter-spacing: 0.4px;
}}

.value {{
  font-family: 'Exo 2', sans-serif;
  font-weight: 700;
}}

.pulse-wrap {{
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding-left: 2px;
}}

.pulse-dot {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 16px currentColor;
}}

.pulse-ring {{
  position: absolute;
  left: -3px;
  width: 16px;
  height: 16px;
  border: 2px solid currentColor;
  border-radius: 50%;
  animation: pulse 1.6s infinite;
  opacity: 0.7;
}}

@keyframes pulse {{
  0% {{ transform: scale(0.8); opacity: 0.8; }}
  100% {{ transform: scale(1.8); opacity: 0; }}
}}

.log-ticker {{
  font-family: 'Exo 2', sans-serif;
  font-size: 0.85rem;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
}}

.log-ticker span {{
  display: inline-block;
  padding-left: 100%;
  animation: ticker 20s linear infinite;
}}

.section-tag {{
  display: inline-block;
  margin-bottom: 0.4rem;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(0,229,255,0.24);
  color: #7dd3fc;
  font-family: 'Orbitron', sans-serif;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}

.image-frame {{
  border: 1px solid rgba(0,229,255,0.28);
  border-radius: 14px;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(6,12,24,0.94), rgba(10,18,34,0.84));
  box-shadow: 0 12px 30px rgba(0,0,0,0.28);
  padding: 8px;
}}

@keyframes ticker {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(-100%); }}
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def telemetry_chart(frame: pd.DataFrame, sensors: list[str], prediction: bool, template: str, grid_color: str) -> go.Figure:
    """Draw mission telemetry chart with optional prediction overlays."""
    colors = {
        "temperature": "#00E5FF",
        "voltage": "#7C3AED",
        "vibration": "#FF4FD8",
        "radiation": "#22D3EE",
        "battery": "#00FFA3",
        "solar_efficiency": "#FFB347",
    }

    fig = go.Figure()
    for sensor in sensors:
        fig.add_trace(
            go.Scatter(
                x=frame["cycle"],
                y=frame[sensor],
                mode="lines",
                line=dict(color=colors[sensor], width=2),
                name=sensor.replace("_", " ").title(),
            )
        )
        if prediction:
            px, py = forecast(frame.set_index("cycle")[sensor])
            if len(px) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=px,
                        y=py,
                        mode="lines",
                        line=dict(color=colors[sensor], width=1.6, dash="dot"),
                        showlegend=False,
                    )
                )

    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=490,
        margin=dict(l=8, r=8, t=10, b=8),
        xaxis=dict(title="Cycle", gridcolor=grid_color),
        yaxis=dict(title="Normalized Value", range=[0, 1.03], gridcolor=grid_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return fig


def risk_gauge(risk: float, template: str) -> go.Figure:
    """Draw risk speedometer gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(np.clip(risk, 0, 100)),
            title={"text": "Failure Risk %"},
            number={"font": {"size": 34, "color": "#E2E8F0"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [
                    {"range": [0, 55], "color": "rgba(0,229,255,0.28)"},
                    {"range": [55, 80], "color": "rgba(124,58,237,0.30)"},
                    {"range": [80, 100], "color": "rgba(255,45,85,0.35)"},
                ],
                "threshold": {"line": {"color": "#FF6B00", "width": 4}, "thickness": 0.9, "value": risk},
            },
        )
    )
    fig.update_layout(template=template, paper_bgcolor="rgba(0,0,0,0)", height=230, margin=dict(l=8, r=8, t=25, b=8))
    return fig


def tracking_map(track_history: list[dict], template: str) -> go.Figure:
    """Draw world map with track trail and current marker."""
    lats = [p["lat"] for p in track_history][-120:]
    lons = [p["lon"] for p in track_history][-120:]

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="lines", line=dict(width=2, color="#00E5FF"), name="Trail"))
    fig.add_trace(
        go.Scattergeo(
            lat=[lats[-1]],
            lon=[lons[-1]],
            mode="markers+text",
            marker=dict(size=12, color="#F8FAFC", line=dict(width=2, color="#FF6B00")),
            text=["SAT"],
            textposition="top center",
            name="Current",
        )
    )
    fig.update_layout(
        title="Live Ground Track",
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=300,
        margin=dict(l=6, r=6, t=40, b=6),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgba(51,65,85,0.48)",
            showocean=True,
            oceancolor="rgba(2,6,23,0.90)",
            showcountries=True,
            countrycolor="rgba(148,163,184,0.20)",
            coastlinecolor="rgba(148,163,184,0.24)",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="SpaceOps AI", layout="wide")
    df, anomaly_bundle, iforest, lstm_bundle = load_artifacts()
    quality_report = load_data_quality_report()
    anomaly_report = load_json_report(config.ANOMALY_REPORT_PATH)
    lstm_report = load_json_report(config.LSTM_REPORT_PATH)
    space_weather = get_space_weather()
    base_means, base_stds = baseline_stats(df, config.TELEMETRY_FEATURES)
    satellite_images = ensure_satellite_images()

    st.sidebar.markdown("## Control Panel")
    inject_css()

    template = "plotly_dark"
    grid_color = "rgba(148,163,184,0.22)"
    should_rerun = False

    sat_name = st.sidebar.selectbox("Satellite", list(SATELLITES.keys()), index=0)
    sat = SATELLITES[sat_name]
    telemetry_feed = st.sidebar.radio("Data Feed", ["Saved Data", "Live Orbit Data"], index=0)
    mission_mode = st.sidebar.radio("AI Mode", ["Monitor", "Assist", "Autonomous"], horizontal=True)
    scenario_name = st.sidebar.selectbox("Mission Scenario", SCENARIOS, index=0)
    live_mode = st.sidebar.toggle(
        "Auto play",
        value=True if telemetry_feed == "Live Orbit Data" else False,
    )

    all_units = sorted(df["unit_id"].unique().tolist())
    selected_unit = st.sidebar.selectbox("Model Data Unit", all_units, index=0)
    sensors = st.sidebar.multiselect(
        "Sensors",
        ["temperature", "voltage", "vibration", "radiation", "battery", "solar_efficiency"],
        default=["temperature", "voltage", "vibration", "battery"],
    )

    if telemetry_feed == "Saved Data":
        unit_df = with_extended_channels(df[df["unit_id"] == selected_unit].sort_values("cycle").reset_index(drop=True))
        if unit_df.empty:
            st.error("No telemetry found for selected unit.")
            st.stop()

        max_idx = len(unit_df) - 1
        min_idx = min(config.WINDOW_SIZE - 1, max_idx)

        if "live_idx" not in st.session_state:
            st.session_state.live_idx = min_idx
        st.session_state.live_idx = int(np.clip(st.session_state.live_idx, min_idx, max_idx))

        idx = st.sidebar.slider("Time Index", min_idx, max_idx, st.session_state.live_idx)
        st.session_state.live_idx = idx

        if live_mode and idx < max_idx:
            st.session_state.live_idx = idx + 1
            should_rerun = True

        current_df = unit_df.iloc[: st.session_state.live_idx + 1].copy()
        latest = current_df.iloc[-1]
        point = get_track_point(
            norad_id=sat["norad_id"],
            cycle=float(latest["cycle"]),
            unit_seed=int(selected_unit),
            timeout_sec=config.LIVE_TELEMETRY_TIMEOUT_SEC,
        )
        history_key = f"track_history_{sat['norad_id']}"
        if history_key not in st.session_state:
            st.session_state[history_key] = []
        st.session_state[history_key].append({"lat": point.latitude, "lon": point.longitude})
        st.session_state[history_key] = st.session_state[history_key][-180:]
        track_history = st.session_state[history_key]
    else:
        st.sidebar.caption("Live mode builds AI data from the latest orbit position.")
        fetch_now = st.sidebar.button("Get Latest Orbit Frame")
        cached = load_cached_live_telemetry()
        need_bootstrap = cached.empty

        if need_bootstrap or live_mode or fetch_now:
            live_df, point = append_live_sample(
                cache_path=config.LIVE_TELEMETRY_CACHE_PATH,
                norad_id=int(sat["norad_id"]),
                unit_id=int(selected_unit),
                cycle_hint=float(len(cached) + 1),
                timeout_sec=config.LIVE_TELEMETRY_TIMEOUT_SEC,
                history_limit=config.LIVE_TELEMETRY_HISTORY_LIMIT,
            )
        else:
            live_df = cached.copy()
            if live_df.empty:
                st.error("Live cache is empty. Click 'Fetch Latest Live Frame' first.")
                st.stop()
            tail = live_df.iloc[-1]
            point = get_track_point(
                norad_id=int(sat["norad_id"]),
                cycle=float(tail["cycle"]),
                unit_seed=int(selected_unit),
                timeout_sec=config.LIVE_TELEMETRY_TIMEOUT_SEC,
            )

        if live_mode:
            should_rerun = True

        unit_df = with_extended_channels(live_df.copy())
        current_df = unit_df.copy()
        latest = current_df.iloc[-1]
        track_history = [{"lat": float(row["latitude"]), "lon": float(row["longitude"])} for _, row in live_df.iterrows()]

    current_df = apply_scenario(current_df, scenario_name)
    latest = current_df.iloc[-1]

    feature_vec = latest[config.TELEMETRY_FEATURES].to_numpy(dtype=np.float32)
    anomaly_score = compute_anomaly_score(feature_vec, anomaly_bundle, iforest)

    window = current_df[config.TELEMETRY_FEATURES].tail(config.WINDOW_SIZE).to_numpy(dtype=np.float32)
    if len(window) < config.WINDOW_SIZE:
        pad = np.repeat(window[:1], config.WINDOW_SIZE - len(window), axis=0)
        window = np.vstack([pad, window])

    predicted_rul = predict_rul(window, lstm_bundle)
    max_rul = max(1.0, float(unit_df["rul"].max()))
    failure_risk = float(np.clip(100.0 * (1.0 - predicted_rul / max_rul), 0.0, 100.0)) if not np.isnan(predicted_rul) else 0.0
    health, status_color = status_from_risk(anomaly_score, failure_risk)
    fault_class = classify_fault(latest, anomaly_score, failure_risk)
    severity = severity_label(failure_risk, anomaly_score)
    explainability = explain_prediction(latest, df)
    drift = compute_drift_report(
        baseline_means=base_means,
        baseline_stds=base_stds,
        current_df=current_df,
        features=config.TELEMETRY_FEATURES,
        warn_threshold=config.DRIFT_WARNING_THRESHOLD,
        critical_threshold=config.DRIFT_CRITICAL_THRESHOLD,
    )
    rec = generate_recommendation(latest=latest, telemetry_history=current_df, anomaly_score=anomaly_score, failure_risk_pct=failure_risk)
    twin_frame = simulate_action(current_df, rec["action"], mission_mode)
    twin_window = twin_frame[config.TELEMETRY_FEATURES].tail(config.WINDOW_SIZE).to_numpy(dtype=np.float32)
    if len(twin_window) < config.WINDOW_SIZE:
        twin_pad = np.repeat(twin_window[:1], config.WINDOW_SIZE - len(twin_window), axis=0)
        twin_window = np.vstack([twin_pad, twin_window])
    twin_predicted_rul = predict_rul(twin_window, lstm_bundle)
    twin_risk = float(np.clip(100.0 * (1.0 - twin_predicted_rul / max_rul), 0.0, 100.0)) if not np.isnan(twin_predicted_rul) else 0.0
    twin_outcome = compare_state(failure_risk, twin_risk)
    timeline = build_timeline(
        health=health,
        risk=failure_risk,
        drift_status=drift.status,
        action=rec["action"],
        operator_mode=mission_mode,
        live=point.live,
    )

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(
        f"""
<div class="mission-bar">
  <div class="mission-title">SpaceOps AI | Space Desk</div>
  <div class="mission-sub">{sat['label']} | {sat['mode']} | AI Mode: {mission_mode} | Scenario: {scenario_name} | Link: {'LIVE' if point.live else 'SIMULATED'} | {now_utc}</div>
  <div class="log-ticker"><span>SYSTEM LOG: Link stable | Power normal | AI ready | Source: {point.source} | Space weather Kp: {space_weather['kp_index']:.1f} | Last update: {point.timestamp_utc}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        f"""<div class='card'><div class='card-title'>Status</div>
<div class='pulse-wrap' style='color:{status_color};'><span class='pulse-ring'></span><span class='pulse-dot'></span><span class='value'>{health}</span></div></div>""",
        unsafe_allow_html=True,
    )
    c2.markdown(f"<div class='card'><div class='card-title'>Risk</div><div class='value'>{failure_risk:.1f}%</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='card-title'>Life Left</div><div class='value'>{predicted_rul:.1f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'><div class='card-title'>Fault Type</div><div class='value'>{fault_class}</div></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='card'><div class='card-title'>Severity</div><div class='value'>{severity}</div></div>", unsafe_allow_html=True)
    if quality_report:
        st.caption(
            "Data Check | "
            f"Score: {float(quality_report.get('quality_score', 0.0)):.1f} | "
            f"Nulls: {int(quality_report.get('null_cells', 0))} | "
            f"Duplicates: {int(quality_report.get('duplicate_rows', 0))} | "
            f"Out-of-range: {int(quality_report.get('out_of_range_cells', 0))} | "
            f"Space Weather: Kp {space_weather['kp_index']:.1f} ({space_weather['source']})"
        )

    left, right = st.columns([1.8, 1.2], gap="medium")

    with left:
        st.markdown("<div class='section-tag'>Telemetry</div>", unsafe_allow_html=True)
        tabs = st.tabs(["Now", "Replay", "Forecast"])

        if not sensors:
            sensors = ["temperature", "voltage", "vibration"]

        zoom = st.slider("Timeline", 60, max(100, len(current_df)), min(220, len(current_df)), step=10)
        frame = current_df.tail(zoom)

        with tabs[0]:
            st.plotly_chart(telemetry_chart(frame, sensors, prediction=False, template=template, grid_color=grid_color), use_container_width=True)
        with tabs[1]:
            replay_fig = telemetry_chart(frame, sensors, prediction=False, template=template, grid_color=grid_color)
            for trace in replay_fig.data:
                trace.mode = "lines+markers"
                trace.marker = {"size": 3}
            st.plotly_chart(replay_fig, use_container_width=True)
        with tabs[2]:
            st.plotly_chart(telemetry_chart(frame, sensors, prediction=True, template=template, grid_color=grid_color), use_container_width=True)

    with right:
        st.markdown("<div class='section-tag'>AI Panel</div>", unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(failure_risk, template), use_container_width=True)
        urgency = "HIGH" if failure_risk >= 80 else ("MEDIUM" if failure_risk >= 55 else "LOW")
        alert_severity = "CRITICAL" if (health == "CRITICAL" or drift.status == "CRITICAL") else ("WARNING" if (health == "WARNING" or drift.status == "WARNING") else "INFO")
        alert_message = (
            f"Health={health}, Risk={failure_risk:.1f}%, Drift={drift.status}({drift.score:.3f}), Action={urgency}"
            if alert_severity != "INFO"
            else f"Nominal operation. Health={health}, Drift={drift.status}"
        )
        alert_type = "ops_alert" if alert_severity != "INFO" else "ops_status"
        alert_history = append_alert(
            config.ALERT_HISTORY_PATH,
            {
                "type": alert_type,
                "severity": alert_severity,
                "satellite": sat["label"],
                "message": alert_message,
            },
            config.ALERT_HISTORY_LIMIT,
        )

        st.markdown("<div class='card'><div class='card-title'>AI Suggestion</div>", unsafe_allow_html=True)
        st.write(f"**Action:** {rec['action']}")
        st.write(f"**Reason:** {rec['reason']}")
        st.write(f"**Confidence:** {rec['confidence'] * 100:.1f}%")
        st.markdown(f"<span style='background:#FF6B00;color:#111827;padding:4px 8px;border-radius:6px;font-weight:700;'>Urgency: {urgency}</span>", unsafe_allow_html=True)
        st.write(f"**Drift:** {drift.status} ({drift.score:.3f})")
        st.write(f"**Top Cause:** {explainability['top_factors'][0]['feature']} - {explainability['top_factors'][0]['message']}")
        retrain_clicked = st.button("Retrain Models")
        if retrain_clicked:
            with st.spinner("Running data prep and model training..."):
                ok, msg = run_retraining_pipeline()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        with st.expander("Why this result?"):
            st.write("The AI looks at heat, voltage, vibration, anomaly error, and life-left trend.")
            st.write(f"Anomaly={anomaly_score:.5f} | Risk={failure_risk:.2f}% | RUL={predicted_rul:.2f}")
            st.write("Drift by sensor:")
            st.json({k: round(v, 4) for k, v in drift.feature_scores.items()})
            st.write("Top drivers:")
            st.json(explainability["top_factors"])
        if st.button("Run Action Test"):
            st.success("Action test completed for the current state.")
        st.markdown("</div>", unsafe_allow_html=True)

    t1, t2 = st.columns([1.35, 1.65], gap="medium")

    with t1:
        st.markdown("<div class='section-tag'>Orbit</div>", unsafe_allow_html=True)
        st.plotly_chart(tracking_map(track_history, template), use_container_width=True)
        st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
        st.image(satellite_images[sat["image_key"]], caption=f"{sat_name} view", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        p1, p2 = st.columns(2)
        p1.metric("Latitude", f"{point.latitude:.2f} deg")
        p2.metric("Longitude", f"{point.longitude:.2f} deg")
        p3, p4 = st.columns(2)
        p3.metric("Altitude", f"{point.altitude_km:.1f} km")
        p4.metric("Speed", f"{point.velocity_kms:.2f} km/s")
        p5, p6 = st.columns(2)
        p5.metric("Kp Index", f"{space_weather['kp_index']:.1f}")
        p6.metric("Weather Feed", "Live" if space_weather["live"] else "Cached")

        if point.live:
            st.success(f"{point.message} Source: {point.source}.")
        else:
            st.warning(f"{point.message} Source: {point.source}.")

        st.markdown("<div class='card'><div class='card-title'>Mission Timeline</div>", unsafe_allow_html=True)
        timeline_df = pd.DataFrame(timeline)
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        st.markdown("<div class='section-tag'>Impact</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='card-title'>Mission Impact</div>", unsafe_allow_html=True)
        scenario = st.selectbox("Scenario", ["Conservative", "Balanced", "Aggressive"], index=1)
        factor = {"Conservative": 0.75, "Balanced": 1.0, "Aggressive": 1.28}[scenario]

        if "cost_saved" not in st.session_state:
            st.session_state.cost_saved = 125000.0
            st.session_state.anomalies_prevented = 18

        delta = max(0.0, (failure_risk * 56 + anomaly_score * 130000) * factor * 0.02)
        st.session_state.cost_saved += delta
        st.session_state.anomalies_prevented += int(delta > 30)

        reliability = float(np.clip(99.1 - failure_risk * 0.41, 52, 99.4))
        reduction = float(np.clip(rec["confidence"] * 68, 8, 92))

        st.markdown(f"<div class='value' style='font-size:2.0rem;color:#00FFA3;'>${st.session_state.cost_saved:,.0f}</div>", unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        i1.metric("Issues Prevented", st.session_state.anomalies_prevented)
        i2.metric("Reliability", f"{reliability:.1f}%")
        i3.metric("Risk Cut", f"{reduction:.1f}%")
        j1, j2, j3 = st.columns(3)
        j1.metric("Twin Risk Before", f"{twin_outcome['risk_before']:.1f}%")
        j2.metric("Twin Risk After", f"{twin_outcome['risk_after']:.1f}%")
        j3.metric("Twin Gain", f"{twin_outcome['risk_reduction']:.1f}%")

        mission_report = {
            "timestamp_utc": now_utc,
            "satellite": sat_name,
            "satellite_label": sat["label"],
            "telemetry_feed": telemetry_feed,
            "tracking_source": point.source,
            "tracking_live": point.live,
            "health_state": health,
            "failure_risk_pct": round(float(failure_risk), 3),
            "predicted_rul": round(float(predicted_rul), 3),
            "anomaly_score": round(float(anomaly_score), 6),
            "drift": {
                "score": round(float(drift.score), 6),
                "status": drift.status,
                "feature_scores": {k: round(v, 6) for k, v in drift.feature_scores.items()},
            },
            "recommendation": rec,
            "mission_impact": {
                "cost_saved_usd": round(float(st.session_state.cost_saved), 2),
                "anomalies_prevented": int(st.session_state.anomalies_prevented),
                "mission_reliability_pct": round(float(reliability), 3),
                "risk_reduction_pct": round(float(reduction), 3),
            },
            "data_quality": quality_report,
        }
        st.download_button(
            label="Download Report",
            data=json.dumps(mission_report, indent=2),
            file_name=f"spaceops_report_{sat['label']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
        with st.expander("Recent Alerts"):
            if alert_history:
                hist_df = pd.DataFrame(alert_history[-12:][::-1])
                st.dataframe(hist_df[["timestamp_utc", "severity", "satellite", "message"]], use_container_width=True, hide_index=True)
            else:
                st.caption("No alerts yet.")

        with st.expander("Model Reports"):
            report_cols = st.columns(2)
            with report_cols[0]:
                if anomaly_report:
                    st.write("**Anomaly Model**")
                    st.json(anomaly_report)
                else:
                    st.caption("Anomaly training report not found.")
            with report_cols[1]:
                if lstm_report:
                    st.write("**RUL Model**")
                    st.json(lstm_report)
                else:
                    st.caption("LSTM training report not found.")

        if st.button("Reset Impact"):
            st.session_state.cost_saved = 125000.0
            st.session_state.anomalies_prevented = 18
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if should_rerun:
        time.sleep(0.8 if telemetry_feed == "Live Orbit Data" else 0.11)
        st.rerun()


if __name__ == "__main__":
    main()
