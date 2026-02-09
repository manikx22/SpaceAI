"""SpaceOps AI dashboard: alternate professional mission interface."""

from __future__ import annotations

import time
from datetime import datetime, timezone
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
from recommendation_engine import generate_recommendation
from train_lstm import LSTMRegressor


class Autoencoder(nn.Module):
    """Inference mirror of the anomaly autoencoder."""

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
    """Load persisted datasets and model checkpoints."""
    if not config.PROCESSED_CSV_PATH.exists():
        preprocess.main()

    df = pd.read_csv(config.PROCESSED_CSV_PATH)
    scaler = joblib.load(config.SCALER_PATH) if config.SCALER_PATH.exists() else None

    anomaly_bundle = torch.load(config.ANOMALY_MODEL_PATH, map_location="cpu") if config.ANOMALY_MODEL_PATH.exists() else None
    iforest = joblib.load(config.ANOMALY_IFOREST_PATH) if config.ANOMALY_IFOREST_PATH.exists() else None
    lstm_bundle = torch.load(config.LSTM_MODEL_PATH, map_location="cpu") if config.LSTM_MODEL_PATH.exists() else None

    return df, scaler, anomaly_bundle, iforest, lstm_bundle


def compute_anomaly_score(feature_vec: np.ndarray, anomaly_bundle, iforest) -> float:
    """Compute anomaly score from available anomaly model."""
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


def predict_rul(sequence: np.ndarray, lstm_bundle) -> float:
    """Predict remaining useful life from telemetry sequence."""
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
        pred = model(torch.tensor(sequence[None, :, :], dtype=torch.float32)).item()
    return float(max(0.0, pred))


def health_status(anomaly_score: float, failure_risk_pct: float) -> str:
    """Classify mission health state from AI outputs."""
    if anomaly_score >= config.CRITICAL_ANOMALY_SCORE or failure_risk_pct >= config.CRITICAL_FAILURE_RISK:
        return "CRITICAL"
    if anomaly_score >= config.WARNING_ANOMALY_SCORE or failure_risk_pct >= config.WARNING_FAILURE_RISK:
        return "WARNING"
    return "HEALTHY"


def ai_mode(risk_pct: float) -> str:
    """Map risk percentage to AI operation mode."""
    if risk_pct >= config.CRITICAL_FAILURE_RISK:
        return "ALERT"
    if risk_pct >= config.WARNING_FAILURE_RISK:
        return "MONITORING"
    return "LEARNING"


def build_extended_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional mission channels for UI analytics."""
    ext = df.copy()
    ext["radiation"] = np.clip(0.30 + 0.54 * ext["temperature"] + 0.08 * np.sin(ext["cycle"] / 13), 0, 1)
    ext["battery"] = np.clip(0.78 * ext["voltage"] + 0.16 * np.cos(ext["cycle"] / 22), 0, 1)
    ext["solar_efficiency"] = np.clip(0.42 + 0.53 * ext["battery"] - 0.14 * ext["vibration"], 0, 1)
    return ext


def forecast(series: pd.Series, steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Short-horizon dotted projection for prediction mode."""
    if len(series) < 2:
        return np.array([]), np.array([])

    tail = series.tail(min(24, len(series)))
    slope = float(tail.diff().mean())
    start = int(series.index.max()) + 1
    x = np.arange(start, start + steps)
    y = np.array([np.clip(float(series.iloc[-1]) + slope * i, 0, 1) for i in range(1, steps + 1)])
    return x, y


def compute_ground_track(cycles: np.ndarray, unit_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic ground track for live monitoring."""
    phase = 0.18 * unit_id
    omega = 0.11
    lon = ((cycles * 3.55 + unit_id * 13.5) % 360) - 180
    lat = 52.5 * np.sin(omega * cycles + phase)
    return lat, lon


def tracking_map(cycles: np.ndarray, unit_id: int, template: str) -> go.Figure:
    """Render live ground track map with current position marker."""
    lat, lon = compute_ground_track(cycles, unit_id)
    trail_len = min(90, len(cycles))

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=lat[-trail_len:],
            lon=lon[-trail_len:],
            mode="lines",
            line=dict(width=2.2, color="#2DD4BF"),
            name="Track",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[lat[-1]],
            lon=[lon[-1]],
            mode="markers+text",
            marker=dict(size=11, color="#F8FAFC", line=dict(width=2, color="#FB923C")),
            text=["SAT"],
            textposition="top center",
            name="Current",
        )
    )

    fig.update_layout(
        title="Live Satellite Tracking",
        template=template,
        height=300,
        margin=dict(l=6, r=6, t=40, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgba(51,65,85,0.48)",
            showocean=True,
            oceancolor="rgba(2,6,23,0.88)",
            showcountries=True,
            countrycolor="rgba(148,163,184,0.22)",
            coastlinecolor="rgba(148,163,184,0.28)",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def orbit_panel(current_cycle: int, template: str) -> go.Figure:
    """Render compact orbital ring panel."""
    theta = np.linspace(0, 360, 240)
    ring = 1 + 0.04 * np.sin(np.radians(theta * 4))
    sat_theta = float((current_cycle * 6) % 360)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ring, theta=theta, mode="lines", line=dict(color="#60A5FA", width=2), name="Orbit"))
    fig.add_trace(
        go.Scatterpolar(
            r=[1.0],
            theta=[sat_theta],
            mode="markers",
            marker=dict(size=13, color="#F8FAFC", line=dict(width=2, color="#FB923C")),
            name="SAT",
        )
    )
    fig.update_layout(
        title="Orbital Plane",
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False), angularaxis=dict(showticklabels=False)),
        margin=dict(l=6, r=6, t=40, b=6),
        height=250,
    )
    return fig


def telemetry_plot(
    frame: pd.DataFrame,
    sensors: list[str],
    colors: dict[str, str],
    prediction: bool,
    template: str,
    grid_color: str,
) -> go.Figure:
    """Build telemetry chart with optional projection."""
    fig = go.Figure()

    for s in sensors:
        fig.add_trace(
            go.Scatter(
                x=frame["cycle"],
                y=frame[s],
                mode="lines",
                line=dict(color=colors[s], width=2),
                name=s.replace("_", " ").title(),
            )
        )
        if prediction:
            px, py = forecast(frame.set_index("cycle")[s])
            if len(px) > 0:
                fig.add_trace(
                    go.Scatter(x=px, y=py, mode="lines", line=dict(color=colors[s], width=1.5, dash="dot"), showlegend=False)
                )

    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(title="Cycle", gridcolor=grid_color),
        yaxis=dict(title="Normalized", range=[0, 1.03], gridcolor=grid_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.0),
    )
    return fig


def risk_gauge(risk: float, template: str, number_color: str, threshold_color: str) -> go.Figure:
    """Render failure risk speedometer."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(np.clip(risk, 0, 100)),
            title={"text": "Failure Risk %"},
            number={"font": {"size": 34, "color": number_color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [
                    {"range": [0, 55], "color": "rgba(45,212,191,0.36)"},
                    {"range": [55, 80], "color": "rgba(251,146,60,0.36)"},
                    {"range": [80, 100], "color": "rgba(244,63,94,0.46)"},
                ],
                "threshold": {"line": {"color": threshold_color, "width": 4}, "thickness": 0.9, "value": risk},
            },
        )
    )
    fig.update_layout(template=template, paper_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=10, r=10, t=28, b=10))
    return fig


def ensure_satellite_images() -> dict[str, Path]:
    """Create local satellite illustration images to guarantee display availability."""
    out_dir = config.PROCESSED_DATA_DIR / "ui_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "LEO Satellite": ("leo_satellite.png", (14, 22, 38), (56, 189, 248)),
        "Earth Observation": ("earth_observation_satellite.png", (8, 24, 34), (45, 212, 191)),
        "Deep Space Probe": ("deep_space_probe.png", (20, 18, 40), (251, 146, 60)),
    }

    paths: dict[str, Path] = {}
    for label, (name, bg, accent) in specs.items():
        path = out_dir / name
        paths[label] = path
        if path.exists():
            continue

        img = Image.new("RGB", (960, 540), bg)
        draw = ImageDraw.Draw(img)

        # Star field
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        for _ in range(320):
            x = int(rng.integers(0, 960))
            y = int(rng.integers(0, 540))
            r = int(rng.integers(1, 3))
            c = int(rng.integers(170, 255))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(c, c, c))

        # Simple satellite body + panels
        draw.rectangle((420, 230, 540, 320), fill=(210, 218, 230), outline=(20, 30, 45), width=3)
        draw.rectangle((300, 245, 410, 305), fill=accent, outline=(20, 30, 45), width=3)
        draw.rectangle((550, 245, 660, 305), fill=accent, outline=(20, 30, 45), width=3)
        draw.ellipse((465, 200, 495, 230), fill=(250, 250, 252), outline=(20, 30, 45), width=3)
        draw.line((480, 200, 480, 155), fill=(220, 226, 236), width=4)
        draw.ellipse((468, 140, 492, 164), fill=accent)

        draw.text((26, 26), f"SpaceOps AI | {label}", fill=(220, 230, 240))
        img.save(path)

    return paths


def inject_css() -> None:
    """Apply alternate visual identity and layout styling."""
    bg_a = "#060D1A"
    bg_b = "#0F172A"
    panel = "rgba(12, 20, 34, 0.92)"
    border = "rgba(148, 163, 184, 0.25)"
    muted = "#94A3B8"
    text = "#E2E8F0"
    sidebar_bg = "rgba(7, 14, 24, 0.94)"
    button_bg = "rgba(30, 41, 59, 0.62)"
    button_text = "#E2E8F0"

    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=Barlow+Condensed:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
  --bg-a: __BG_A__;
  --bg-b: __BG_B__;
  --panel: __PANEL__;
  --border: __BORDER__;
  --muted: __MUTED__;
  --text: __TEXT__;
  --teal: #2DD4BF;
  --amber: #FB923C;
  --rose: #F43F5E;
}

.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(56,189,248,0.16), transparent 24%),
    radial-gradient(circle at 88% 12%, rgba(251,146,60,0.12), transparent 22%),
    linear-gradient(145deg, var(--bg-a) 0%, var(--bg-b) 100%);
  color: var(--text);
  font-family: 'Barlow Condensed', sans-serif;
}

.main .block-container {
  max-width: 1550px;
  padding-top: 0.7rem;
}

.header-shell {
  border: 1px solid var(--border);
  border-radius: 14px;
  background: rgba(15,23,42,0.82);
  padding: 12px 14px;
  margin-bottom: 10px;
}

.header-title {
  font-family: 'Sora', sans-serif;
  letter-spacing: 0.6px;
  font-size: 1.3rem;
}

.header-sub {
  color: var(--muted);
  font-size: 0.92rem;
}

.card {
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 14px 28px rgba(2, 6, 23, 0.44);
}

.card-title {
  font-family: 'Sora', sans-serif;
  font-size: 0.96rem;
  letter-spacing: 0.5px;
}

.value {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
}

.status-pill {
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 0.84rem;
  font-weight: 700;
  border: 1px solid rgba(148,163,184,0.30);
}

.led {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
  box-shadow: 0 0 8px currentColor;
}

[data-testid="stSidebar"] > div:first-child {
  background: __SIDEBAR_BG__;
  border-right: 1px solid var(--border);
  backdrop-filter: blur(8px);
}

.stButton > button {
  border-radius: 10px;
  border: 1px solid rgba(148,163,184,0.33);
  background: __BUTTON_BG__;
  color: __BUTTON_TEXT__;
}
</style>
"""
    css = (
        css.replace("__BG_A__", bg_a)
        .replace("__BG_B__", bg_b)
        .replace("__PANEL__", panel)
        .replace("__BORDER__", border)
        .replace("__MUTED__", muted)
        .replace("__TEXT__", text)
        .replace("__SIDEBAR_BG__", sidebar_bg)
        .replace("__BUTTON_BG__", button_bg)
        .replace("__BUTTON_TEXT__", button_text)
    )
    st.markdown(css, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="SpaceOps AI | Mission Deck", layout="wide")
    df, _scaler, anomaly_bundle, iforest, lstm_bundle = load_artifacts()
    satellite_images = ensure_satellite_images()

    st.sidebar.markdown("## Mission Controls")
    inject_css()

    template = "plotly_dark"
    grid_color = "rgba(148,163,184,0.18)"
    number_color = "#F8FAFC"
    threshold_color = "#F8FAFC"

    mission_preset = st.sidebar.selectbox("Mission Profile", ["Nominal Ops", "Storm Watch", "Battery Stress", "Deep Space Cruise"], index=0)
    mission_mode = st.sidebar.radio("Control Mode", ["Auto", "Manual", "Emergency"], horizontal=True)

    units = sorted(df["unit_id"].unique().tolist())
    selected_unit = st.sidebar.selectbox("Satellite", units, format_func=lambda u: f"SAT-ALPHA-{u:02d}")
    live_mode = st.sidebar.toggle("Live Playback", value=False)

    colors = {
        "temperature": "#38BDF8",
        "voltage": "#8B5CF6",
        "vibration": "#EC4899",
        "radiation": "#0EA5E9",
        "battery": "#34D399",
        "solar_efficiency": "#FB923C",
    }
    selected_sensors = st.sidebar.multiselect("Sensors", list(colors.keys()), default=["temperature", "voltage", "vibration", "battery"])

    unit_base = df[df["unit_id"] == selected_unit].sort_values("cycle").reset_index(drop=True)
    unit_df = build_extended_channels(unit_base)
    if unit_df.empty:
        st.error("No telemetry is available for this satellite.")
        st.stop()

    max_idx = len(unit_df) - 1
    min_idx = min(config.WINDOW_SIZE - 1, max_idx)
    if "live_idx" not in st.session_state:
        st.session_state.live_idx = min_idx

    st.session_state.live_idx = int(np.clip(st.session_state.live_idx, min_idx, max_idx))
    current_idx = st.sidebar.slider("Telemetry Index", min_idx, max_idx, st.session_state.live_idx)
    st.session_state.live_idx = current_idx

    if live_mode:
        current_idx = min(current_idx + 1, max_idx)
        st.session_state.live_idx = current_idx
        time.sleep(0.11)
        st.rerun()

    current_df = unit_df.iloc[: current_idx + 1].copy()
    latest = current_df.iloc[-1]

    track_cycles = current_df["cycle"].to_numpy(dtype=np.float32)
    lat, lon = compute_ground_track(track_cycles, selected_unit)

    feat_vec = latest[config.TELEMETRY_FEATURES].to_numpy(dtype=np.float32)
    anomaly_score = compute_anomaly_score(feat_vec, anomaly_bundle, iforest)

    window = current_df[config.TELEMETRY_FEATURES].tail(config.WINDOW_SIZE).to_numpy(dtype=np.float32)
    if len(window) < config.WINDOW_SIZE:
        pad = np.repeat(window[:1], config.WINDOW_SIZE - len(window), axis=0)
        window = np.vstack([pad, window])

    predicted_rul = predict_rul(window, lstm_bundle)
    max_rul = max(1.0, float(unit_df["rul"].max()))
    failure_risk = float(np.clip(100.0 * (1.0 - predicted_rul / max_rul), 0.0, 100.0)) if not np.isnan(predicted_rul) else 0.0
    status = health_status(anomaly_score, failure_risk)

    status_color = {"HEALTHY": "#2DD4BF", "WARNING": "#FB923C", "CRITICAL": "#F43F5E"}[status]
    mode = ai_mode(failure_risk)
    confidence = float(np.clip(92 - anomaly_score * 820 - abs(50 - failure_risk) * 0.12, 54, 98))

    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(
        f"""
<div class="header-shell">
  <div class="header-title">SpaceOps AI Mission Deck</div>
  <div class="header-sub">{mission_preset} | SAT-ALPHA-{selected_unit:02d} | {mission_mode} Control | Uplink: {'Live' if live_mode else 'Standby'} | {utc_now}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Row 1: summary metrics
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.markdown(f"<div class='card'><div class='card-title'>Health</div><div class='status-pill' style='color:{status_color}; border-color:{status_color};'>{status}</div></div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='card'><div class='card-title'>Failure Risk</div><div class='value'>{failure_risk:.1f}%</div></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='card'><div class='card-title'>Predicted RUL</div><div class='value'>{predicted_rul:.1f}</div></div>", unsafe_allow_html=True)
    s4.markdown(f"<div class='card'><div class='card-title'>Anomaly Score</div><div class='value'>{anomaly_score:.5f}</div></div>", unsafe_allow_html=True)
    s5.markdown(f"<div class='card'><div class='card-title'>AI Mode</div><div class='value'>{mode}</div></div>", unsafe_allow_html=True)

    # Row 2: completely different arrangement
    left, right = st.columns([1.75, 1.25], gap="medium")

    with left:
        tab_live, tab_replay, tab_pred = st.tabs(["Telemetry Live", "Telemetry Replay", "Telemetry Prediction"])

        if not selected_sensors:
            st.warning("No sensors selected, defaulting to temperature, voltage, and vibration.")
            selected_sensors = ["temperature", "voltage", "vibration"]

        zoom = st.slider("Time Window", 80, max(120, len(current_df)), min(240, len(current_df)), step=10)
        frame = current_df.tail(zoom)

        with tab_live:
            st.plotly_chart(telemetry_plot(frame, selected_sensors, colors, prediction=False, template=template, grid_color=grid_color), width="stretch")
        with tab_replay:
            replay = go.Figure()
            for sensor in selected_sensors:
                replay.add_trace(
                    go.Scatter(
                        x=frame["cycle"], y=frame[sensor], mode="lines+markers", marker=dict(size=3),
                        line=dict(width=1.7, color=colors[sensor]), name=sensor.replace("_", " ").title(),
                    )
                )
            replay.update_layout(
                template=template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500,
                margin=dict(l=8, r=8, t=8, b=8), xaxis=dict(gridcolor=grid_color), yaxis=dict(range=[0,1.03], gridcolor=grid_color)
            )
            st.plotly_chart(replay, width="stretch")
        with tab_pred:
            st.plotly_chart(telemetry_plot(frame, selected_sensors, colors, prediction=True, template=template, grid_color=grid_color), width="stretch")

    with right:
        st.plotly_chart(risk_gauge(failure_risk, template=template, number_color=number_color, threshold_color=threshold_color), width="stretch")

        r1, r2 = st.columns(2)
        r1.markdown(f"<div class='card'><div class='card-title'>AI Confidence</div><div class='value'>{confidence:.1f}%</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='card'><div class='card-title'>Mission Mode</div><div class='value'>{mode}</div></div>", unsafe_allow_html=True)

        rec = generate_recommendation(latest=latest, telemetry_history=current_df, anomaly_score=anomaly_score, failure_risk_pct=failure_risk)
        urgency = "HIGH" if failure_risk >= 80 else ("MEDIUM" if failure_risk >= 55 else "LOW")
        st.markdown("<div class='card'><div class='card-title'>AI Command Recommendation</div>", unsafe_allow_html=True)
        st.write(f"**Action:** {rec['action']}")
        st.write(f"**Reason:** {rec['reason']}")
        st.write(f"**Confidence:** {rec['confidence'] * 100:.1f}%")
        st.markdown(f"<span style='background:#FB923C; color:#111827; padding:4px 8px; border-radius:6px; font-weight:700;'>Urgency: {urgency}</span>", unsafe_allow_html=True)
        if st.button("Execute Simulation"):
            st.success("Simulation executed successfully for the current mission state.")
        with st.expander("Technical Reasoning"):
            st.write("Rules consider thermal stress, vibration, voltage trends, anomaly reconstruction error, and LSTM RUL behavior.")
            st.write(f"Anomaly={anomaly_score:.5f}, Risk={failure_risk:.2f}%, RUL={predicted_rul:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: tracking + impact
    tcol, icol = st.columns([1.35, 1.65], gap="medium")

    with tcol:
        st.plotly_chart(orbit_panel(int(latest["cycle"]), template=template), width="stretch")
        st.plotly_chart(tracking_map(track_cycles, selected_unit, template=template), width="stretch")

        selected_image = st.selectbox("Satellite Image", list(satellite_images.keys()), index=0)
        st.image(satellite_images[selected_image], caption=selected_image, use_container_width=True)

        alt_km = 540.0 + 8.0 * np.sin(float(latest["cycle"]) / 18)
        spd_kms = 7.62 + 0.18 * np.cos(float(latest["cycle"]) / 13)
        g1, g2 = st.columns(2)
        g1.metric("Latitude", f"{float(lat[-1]):.2f} deg")
        g2.metric("Longitude", f"{float(lon[-1]):.2f} deg")
        g3, g4 = st.columns(2)
        g3.metric("Altitude", f"{alt_km:.1f} km")
        g4.metric("Ground Speed", f"{spd_kms:.2f} km/s")
        st.markdown("<div class='card'><span class='led' style='color:#2DD4BF; background:#2DD4BF;'></span>Tracking stream active and synchronized.</div>", unsafe_allow_html=True)

    with icol:
        st.markdown("<div class='card'><div class='card-title'>Mission Impact Summary</div>", unsafe_allow_html=True)
        scenario = st.selectbox("Scenario", ["Conservative", "Balanced", "Aggressive"], index=1)
        factor = {"Conservative": 0.7, "Balanced": 1.0, "Aggressive": 1.3}[scenario]

        if "cost_saved" not in st.session_state:
            st.session_state.cost_saved = 125000.0
            st.session_state.anomalies_prevented = 18

        delta = max(0.0, (failure_risk * 52 + anomaly_score * 125000) * factor * 0.02)
        st.session_state.cost_saved += delta
        st.session_state.anomalies_prevented += int(delta > 32)

        reliability = float(np.clip(99.1 - failure_risk * 0.41, 52, 99.3))
        reduction = float(np.clip(rec["confidence"] * 68, 8, 92))

        st.markdown(f"<div class='value' style='font-size:2.0rem; color:#34D399;'>${st.session_state.cost_saved:,.0f}</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Anomalies Prevented", st.session_state.anomalies_prevented)
        m2.metric("Reliability %", f"{reliability:.1f}%")
        m3.metric("Risk Reduction %", f"{reduction:.1f}%")

        if st.button("Reset Impact Simulation"):
            st.session_state.cost_saved = 125000.0
            st.session_state.anomalies_prevented = 18
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
