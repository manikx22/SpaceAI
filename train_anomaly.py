"""Train anomaly detector (Autoencoder preferred, IsolationForest fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
import joblib

import config
import preprocess
from utils.io_utils import ensure_dirs, exists_all


class Autoencoder(nn.Module):
    """Simple dense autoencoder for telemetry reconstruction."""

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
        z = self.encoder(x)
        return self.decoder(z)


def _ensure_processed_data() -> pd.DataFrame:
    required = [config.PROCESSED_CSV_PATH, config.SCALER_PATH]
    if not exists_all(required):
        print("Processed data missing. Running preprocess.py...")
        preprocess.main()
    return pd.read_csv(config.PROCESSED_CSV_PATH)


def train_autoencoder(df: pd.DataFrame) -> None:
    """Train PyTorch autoencoder and save state dict with threshold."""
    features = df[config.TELEMETRY_FEATURES].to_numpy(dtype=np.float32)
    tensor_x = torch.tensor(features)

    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=config.ANOMALY_BATCH_SIZE, shuffle=True)

    model = Autoencoder(input_dim=features.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ANOMALY_LR)

    model.train()
    for epoch in range(config.ANOMALY_EPOCHS):
        epoch_losses = []
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f"[Autoencoder] Epoch {epoch + 1}/{config.ANOMALY_EPOCHS} - loss={np.mean(epoch_losses):.6f}")

    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_x)
        errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()

    threshold = float(np.percentile(errors, config.ANOMALY_THRESHOLD_PERCENTILE))

    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": features.shape[1],
        "features": config.TELEMETRY_FEATURES,
        "threshold": threshold,
    }
    torch.save(payload, config.ANOMALY_MODEL_PATH)
    print(f"Saved autoencoder model to {config.ANOMALY_MODEL_PATH}")
    print(f"Anomaly threshold (p{config.ANOMALY_THRESHOLD_PERCENTILE}): {threshold:.6f}")


def train_isolation_forest(df: pd.DataFrame) -> None:
    """Train IsolationForest as robust fallback."""
    features = df[config.TELEMETRY_FEATURES].to_numpy(dtype=np.float32)
    model = IsolationForest(n_estimators=250, contamination=0.05, random_state=config.RANDOM_SEED)
    model.fit(features)
    joblib.dump(model, config.ANOMALY_IFOREST_PATH)
    print(f"Saved IsolationForest fallback to {config.ANOMALY_IFOREST_PATH}")


def main() -> None:
    ensure_dirs([config.MODELS_DIR])
    df = _ensure_processed_data()

    try:
        train_autoencoder(df)
    except Exception as exc:
        print(f"Autoencoder training failed: {exc}")
        print("Falling back to IsolationForest...")
        train_isolation_forest(df)

    # Always keep fallback model available for production resiliency.
    if not config.ANOMALY_IFOREST_PATH.exists():
        train_isolation_forest(df)


if __name__ == "__main__":
    main()
