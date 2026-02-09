# SpaceOps AI

SpaceOps AI is an end-to-end AI system that predicts satellite-like faults from telemetry time-series data and recommends self-healing actions through a Streamlit dashboard.

## Overview

The project uses NASA CMAPSS turbofan data (or synthetic fallback telemetry) as a proxy for satellite subsystem telemetry.

Core capabilities:
- Data pipeline: load, normalize, label RUL, generate LSTM windows
- Anomaly detection: PyTorch autoencoder with IsolationForest fallback
- Failure prediction: PyTorch LSTM for Remaining Useful Life (RUL)
- Recommendation engine: rule-based corrective actions with confidence
- Dashboard: live telemetry visualization, anomaly alerts, failure risk, recommendations, and simulated mission cost savings

## Architecture

- `preprocess.py`: data ingestion, synthetic fallback generation, MinMax scaling, RUL/sequence creation
- `train_anomaly.py`: autoencoder anomaly model training and fallback isolation forest
- `train_lstm.py`: LSTM RUL regression training and evaluation metrics
- `recommendation_engine.py`: self-healing rule evaluation
- `app.py`: Streamlit UI and real-time inference panel
- `config.py`: central paths, thresholds, and hyperparameters

## Project Structure

```
spaceops_ai/
├ data/
│  ├ raw/
│  └ processed/
├ models/
├ training/
├ dashboard/
├ utils/
├ app.py
├ train_anomaly.py
├ train_lstm.py
├ preprocess.py
├ recommendation_engine.py
├ config.py
├ requirements.txt
└ README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python preprocess.py
python train_anomaly.py
python train_lstm.py
```

Notes:
- To use NASA CMAPSS real data, place `train_FD001.txt` in `data/raw/`.
- If CMAPSS is missing, synthetic telemetry is generated automatically.

## Launch Dashboard

```bash
streamlit run app.py
```

## Output Artifacts

- Processed data: `data/processed/telemetry_processed.csv`
- Scaler: `data/processed/telemetry_scaler.joblib`
- LSTM windows: `data/processed/lstm_X.npy`, `data/processed/lstm_y.npy`
- Anomaly model: `models/anomaly_autoencoder.pt`
- Fallback anomaly model: `models/anomaly_iforest.joblib`
- Failure model: `models/lstm_failure_model.pt`

## Expected Dashboard Panels

1. Satellite Status (`HEALTHY / WARNING / CRITICAL`)
2. Telemetry Visualization (Temperature, Voltage, Vibration)
3. AI Prediction Panel (Failure Risk, Predicted RUL, Anomaly Score)
4. Recommendation Panel (Action + Confidence)
5. Mission Cost Saved (simulated counter)
