"""Scenario presets for mission condition simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCENARIOS = [
    "Nominal",
    "Solar Storm",
    "Battery Drain",
    "Thermal Spike",
    "Comms Noise",
]


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Apply a mission scenario to telemetry without breaking model feature bounds."""
    frame = df.copy()
    if frame.empty or scenario == "Nominal":
        return frame

    if scenario == "Solar Storm":
        frame["temperature"] = np.clip(frame["temperature"] + 0.08, 0, 1)
        frame["voltage"] = np.clip(frame["voltage"] - 0.05, 0, 1)
        frame["vibration"] = np.clip(frame["vibration"] + 0.03, 0, 1)
    elif scenario == "Battery Drain":
        frame["voltage"] = np.clip(frame["voltage"] - 0.12, 0, 1)
        frame["fuel_flow"] = np.clip(frame["fuel_flow"] + 0.06, 0, 1)
        frame["rpm"] = np.clip(frame["rpm"] - 0.05, 0, 1)
    elif scenario == "Thermal Spike":
        frame["temperature"] = np.clip(frame["temperature"] + 0.14, 0, 1)
        frame["pressure"] = np.clip(frame["pressure"] + 0.06, 0, 1)
    elif scenario == "Comms Noise":
        oscillation = 0.04 * np.sin(np.arange(len(frame)) / 3.0)
        frame["vibration"] = np.clip(frame["vibration"] + oscillation, 0, 1)
        frame["voltage"] = np.clip(frame["voltage"] - np.abs(oscillation) * 0.5, 0, 1)

    return frame
