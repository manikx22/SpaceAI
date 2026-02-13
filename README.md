# SpaceOps AI

SpaceOps AI is a hackathon-ready AI mission assistant that predicts satellite-like faults from telemetry time series and recommends corrective actions in a live Streamlit mission dashboard.

## Why This Project

Satellite operations teams need early warning and clear actions, not just raw telemetry. SpaceOps AI combines:
- Anomaly detection (Autoencoder + Isolation Forest fallback)
- Failure progression modeling (LSTM RUL prediction)
- Rule-based recommendation logic
- A mission-control style dashboard with live tracking

## Key Features

- CMAPSS-compatible preprocessing pipeline with synthetic fallback
- RUL label generation + sliding window sequence builder
- PyTorch anomaly autoencoder training pipeline
- PyTorch LSTM failure predictor with validation metrics
- Streamlit dashboard with:
  - Health status and AI risk panel
  - Live/Replay/Prediction telemetry tabs
  - Recommendation console with urgency tag
  - Mission impact simulation
  - Live satellite tracking (Real TLE via CelesTrak + simulated fallback)
- NORAD satellite selector for real orbital position
- Local satellite images for offline-safe UI rendering
- Dark mode optimized UI

## Tech Stack

- Python, Pandas, NumPy, scikit-learn
- PyTorch
- Streamlit + Plotly
- Joblib

## Repository Structure

```text
.
├── .github/
│   ├── workflows/
│   └── ISSUE_TEMPLATE/
├── data/
│   ├── raw/
│   └── processed/          # runtime artifacts (ignored)
├── docs/
├── models/                 # runtime artifacts (ignored)
├── tests/
├── utils/
├── app.py
├── config.py
├── preprocess.py
├── recommendation_engine.py
├── train_anomaly.py
├── train_lstm.py
├── requirements.txt
└── README.md
```

## Quick Start

### 1) Install

```bash
python -m pip install -r requirements.txt
```

### 2) Train models

```bash
python preprocess.py
python train_anomaly.py
python train_lstm.py
```

### 3) Run dashboard

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open: [http://127.0.0.1:8501](http://127.0.0.1:8501)


## Real Satellite Tracking

The dashboard supports two tracking modes in the sidebar:
- `Real TLE (CelesTrak)`: Fetches live TLE for a NORAD ID and computes current subpoint using Skyfield
- `Simulated`: Deterministic fallback track for offline/demo reliability

Default NORAD ID is `25544` (ISS).

## Data

Default behavior:
- Looks for CMAPSS train file in `data/raw/`
- If missing, generates realistic synthetic telemetry automatically

Supported CMAPSS filenames:
- `data/raw/train_FD001.txt`
- `data/raw/FD001.txt`
- `data/raw/CMAPSSData/train_FD001.txt`

## Model Outputs

Generated artifacts:
- `data/processed/telemetry_processed.csv`
- `data/processed/telemetry_scaler.joblib`
- `data/processed/lstm_X.npy`
- `data/processed/lstm_y.npy`
- `models/anomaly_autoencoder.pt`
- `models/anomaly_iforest.joblib`
- `models/lstm_failure_model.pt`

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) validates:
- Python setup
- Dependency install
- Syntax compile check
- Unit tests

## Hackathon Notes

- Runtime artifacts (`data/processed`, `models`) are intentionally gitignored.
- Dashboard satellite images are generated locally in `data/processed/ui_assets/` at runtime.
- Live tracking currently uses a deterministic synthetic orbital track model for demo consistency.

## Future Improvements

- Real orbital propagation (TLE + SGP4/Skyfield)
- Better LSTM feature engineering and sequence split strategy
- Containerized deployment (Docker)
- Experiment tracking (MLflow/W&B)

## License

This project is released under the MIT License. See `LICENSE`.
