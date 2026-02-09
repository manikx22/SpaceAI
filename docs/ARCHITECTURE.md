# Architecture

## Pipeline

1. `preprocess.py`
- Load CMAPSS (or synthesize telemetry)
- Normalize telemetry features
- Generate RUL labels
- Build LSTM sliding windows
- Save processed artifacts

2. `train_anomaly.py`
- Train dense autoencoder on telemetry vectors
- Compute reconstruction-error threshold
- Save autoencoder checkpoint
- Train/save IsolationForest fallback

3. `train_lstm.py`
- Load sequence windows
- Train LSTM regressor for RUL
- Compute validation metrics
- Save model checkpoint

4. `app.py`
- Load models and processed data
- Infer anomaly score + RUL + risk
- Render mission dashboard and tracking map
- Generate rule-based recommendations

## Safety and Fallbacks

- Missing CMAPSS dataset => synthetic telemetry generation
- Missing anomaly autoencoder => fallback to IsolationForest
- Missing LSTM model => RUL displays as unavailable
