"""Central configuration for SpaceOps AI."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# NASA CMAPSS raw file candidates (drop one in data/raw to use real data)
CMAPSS_CANDIDATES = [
    RAW_DATA_DIR / "train_FD001.txt",
    RAW_DATA_DIR / "FD001.txt",
    RAW_DATA_DIR / "CMAPSSData" / "train_FD001.txt",
]

# Processed artifacts
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "telemetry_processed.csv"
SCALER_PATH = PROCESSED_DATA_DIR / "telemetry_scaler.joblib"
LSTM_X_PATH = PROCESSED_DATA_DIR / "lstm_X.npy"
LSTM_Y_PATH = PROCESSED_DATA_DIR / "lstm_y.npy"
UNIT_INDEX_PATH = PROCESSED_DATA_DIR / "unit_index.npy"
LIVE_TELEMETRY_CACHE_PATH = PROCESSED_DATA_DIR / "live_telemetry_cache.json"
DATA_QUALITY_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.json"
ALERT_HISTORY_PATH = PROCESSED_DATA_DIR / "alert_history.json"
ANOMALY_REPORT_PATH = PROCESSED_DATA_DIR / "anomaly_training_report.json"
LSTM_REPORT_PATH = PROCESSED_DATA_DIR / "lstm_training_report.json"
SPACE_WEATHER_CACHE_PATH = PROCESSED_DATA_DIR / "space_weather_cache.json"

# Model artifacts
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_autoencoder.pt"
ANOMALY_IFOREST_PATH = MODELS_DIR / "anomaly_iforest.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_failure_model.pt"

# Features used by both models and UI
TELEMETRY_FEATURES = [
    "temperature",
    "voltage",
    "vibration",
    "pressure",
    "fuel_flow",
    "rpm",
]

# Preprocessing/training hyperparameters
WINDOW_SIZE = 50
RANDOM_SEED = 42

ANOMALY_EPOCHS = 20
ANOMALY_BATCH_SIZE = 256
ANOMALY_LR = 1e-3
ANOMALY_THRESHOLD_PERCENTILE = 95

LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 128
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_LR = 1e-3

# Rule thresholds (in normalized [0,1] scale because of MinMaxScaler)
TEMP_THRESHOLD = 0.8
VIBRATION_THRESHOLD = 0.75
VOLTAGE_LOW_THRESHOLD = 0.25
VOLTAGE_DROP_TREND = -0.003

# Dashboard thresholds
WARNING_FAILURE_RISK = 55.0
CRITICAL_FAILURE_RISK = 80.0
WARNING_ANOMALY_SCORE = 0.025
CRITICAL_ANOMALY_SCORE = 0.05

# Live telemetry settings
LIVE_TELEMETRY_TIMEOUT_SEC = 6.0
LIVE_TELEMETRY_HISTORY_LIMIT = 240

# Drift/alerts
DRIFT_WARNING_THRESHOLD = 0.18
DRIFT_CRITICAL_THRESHOLD = 0.30
ALERT_HISTORY_LIMIT = 300

# Explainability / digital twin
TOP_EXPLANATION_FACTORS = 3
