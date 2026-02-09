"""Preprocess telemetry data and create LSTM windows for SpaceOps AI."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

import config
from utils.io_utils import ensure_dirs


def _find_cmapss_file() -> Path | None:
    for candidate in config.CMAPSS_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _load_cmapss(path: Path) -> pd.DataFrame:
    """Load CMAPSS train file and map sensor channels to telemetry columns."""
    columns = ["unit_id", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    # Some CMAPSS files include trailing empty columns from whitespace.
    raw = raw.iloc[:, : len(columns)]
    raw.columns = columns

    max_cycle = raw.groupby("unit_id")["cycle"].transform("max")
    raw["rul"] = max_cycle - raw["cycle"]

    telemetry = pd.DataFrame(
        {
            "unit_id": raw["unit_id"],
            "cycle": raw["cycle"],
            "temperature": raw["sensor_11"],
            "voltage": raw["sensor_7"],
            "vibration": raw["sensor_12"],
            "pressure": raw["sensor_4"],
            "fuel_flow": raw["sensor_15"],
            "rpm": raw["sensor_21"],
            "rul": raw["rul"],
        }
    )
    return telemetry


def _generate_synthetic_data(num_units: int = 40, min_cycles: int = 140, max_cycles: int = 300) -> pd.DataFrame:
    """Generate realistic telemetry when CMAPSS is unavailable."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for unit_id in range(1, num_units + 1):
        total_cycles = int(rng.integers(min_cycles, max_cycles))
        base_temp = rng.uniform(0.45, 0.6)
        base_voltage = rng.uniform(0.65, 0.8)
        base_vibration = rng.uniform(0.25, 0.4)

        for cycle in range(1, total_cycles + 1):
            progress = cycle / total_cycles
            degradation = progress**1.8

            temperature = base_temp + 0.32 * degradation + rng.normal(0, 0.015)
            voltage = base_voltage - 0.30 * degradation + rng.normal(0, 0.010)
            vibration = base_vibration + 0.40 * degradation + rng.normal(0, 0.020)
            pressure = 0.55 + 0.15 * np.sin(cycle / 18) + 0.2 * degradation + rng.normal(0, 0.02)
            fuel_flow = 0.50 + 0.18 * degradation + rng.normal(0, 0.02)
            rpm = 0.58 - 0.20 * degradation + rng.normal(0, 0.02)

            rows.append(
                {
                    "unit_id": unit_id,
                    "cycle": cycle,
                    "temperature": temperature,
                    "voltage": voltage,
                    "vibration": vibration,
                    "pressure": pressure,
                    "fuel_flow": fuel_flow,
                    "rpm": rpm,
                    "rul": total_cycles - cycle,
                }
            )

    return pd.DataFrame(rows)


def _create_windows(df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows grouped by unit to prevent sequence leakage."""
    sequences = []
    targets = []
    units = []
    feat_cols = config.TELEMETRY_FEATURES

    for unit_id, unit_df in df.groupby("unit_id"):
        unit_df = unit_df.sort_values("cycle")
        values = unit_df[feat_cols].to_numpy(dtype=np.float32)
        rul = unit_df["rul"].to_numpy(dtype=np.float32)

        if len(unit_df) < window_size:
            continue

        for idx in range(window_size - 1, len(unit_df)):
            start = idx - window_size + 1
            sequences.append(values[start : idx + 1])
            targets.append(rul[idx])
            units.append(unit_id)

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(units, dtype=np.int32)


def main() -> None:
    """Run complete preprocessing and save processed artifacts."""
    ensure_dirs([config.DATA_DIR, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.MODELS_DIR])

    cmapss_path = _find_cmapss_file()
    if cmapss_path:
        telemetry = _load_cmapss(cmapss_path)
        print(f"Loaded CMAPSS dataset from {cmapss_path}")
    else:
        telemetry = _generate_synthetic_data()
        print("CMAPSS dataset not found. Generated synthetic telemetry data.")

    telemetry = telemetry.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    scaler = MinMaxScaler()
    telemetry[config.TELEMETRY_FEATURES] = scaler.fit_transform(telemetry[config.TELEMETRY_FEATURES])

    X, y, units = _create_windows(telemetry, config.WINDOW_SIZE)

    telemetry.to_csv(config.PROCESSED_CSV_PATH, index=False)
    joblib.dump(scaler, config.SCALER_PATH)
    np.save(config.LSTM_X_PATH, X)
    np.save(config.LSTM_Y_PATH, y)
    np.save(config.UNIT_INDEX_PATH, units)

    print(f"Saved processed telemetry: {config.PROCESSED_CSV_PATH}")
    print(f"Saved scaler: {config.SCALER_PATH}")
    print(f"Saved LSTM windows: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
