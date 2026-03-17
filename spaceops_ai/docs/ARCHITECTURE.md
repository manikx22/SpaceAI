# Architecture

SpaceOps AI is structured as a mission-operations pipeline rather than a single-model demo.

## Flow

1. Telemetry enters from NASA CMAPSS proxy data or live orbit-derived feeds.
2. `preprocess.py` normalizes features, generates RUL labels, builds windows, and writes quality reports.
3. `train_anomaly.py` builds the anomaly detector and persists a training report.
4. `train_lstm.py` builds the failure forecasting model and persists validation metrics.
5. Shared inference in `utils/inference.py` powers both the Streamlit dashboard and FastAPI service.
6. Explainability, drift monitoring, timeline generation, digital twin simulation, and alert history enrich the raw prediction into an operations decision layer.

## Components

- `app.py`: primary mission-control UI
- `api.py`: programmatic inference and simulation surface
- `utils/explainability.py`: interpretable fault drivers
- `utils/digital_twin.py`: recommended action outcome simulation
- `utils/model_monitoring.py`: drift monitoring
- `utils/mission_timeline.py`: replay-style incident sequencing
- `utils/space_weather.py`: external mission context

## Design Goal

The system is designed around the core problem of satellite fault prediction and recovery support:

- detect faults early
- forecast failure progression
- explain the likely cause
- recommend a response
- estimate the effect of that response before execution
