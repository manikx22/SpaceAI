"""Train LSTM model for Remaining Useful Life (RUL) prediction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
import preprocess
from utils.io_utils import ensure_dirs, exists_all


class LSTMRegressor(nn.Module):
    """Sequence model that predicts RUL from telemetry windows."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _ensure_sequence_data() -> tuple[np.ndarray, np.ndarray]:
    required = [config.LSTM_X_PATH, config.LSTM_Y_PATH]
    if not exists_all(required):
        print("LSTM sequences missing. Running preprocess.py...")
        preprocess.main()
    X = np.load(config.LSTM_X_PATH)
    y = np.load(config.LSTM_Y_PATH)
    return X, y


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8)
    r2 = float(1.0 - (np.sum(err**2) / denom))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    ensure_dirs([config.MODELS_DIR])
    X, y = _ensure_sequence_data()

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=config.LSTM_BATCH_SIZE,
        shuffle=True,
    )

    model = LSTMRegressor(
        input_size=X.shape[-1],
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_LR)

    for epoch in range(config.LSTM_EPOCHS):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"[LSTM] Epoch {epoch + 1}/{config.LSTM_EPOCHS} - loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.tensor(X_val, dtype=torch.float32)).numpy()

    metrics = _regression_metrics(y_val, val_pred)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": X.shape[-1],
        "hidden_size": config.LSTM_HIDDEN_SIZE,
        "num_layers": config.LSTM_NUM_LAYERS,
        "window_size": config.WINDOW_SIZE,
        "features": config.TELEMETRY_FEATURES,
        "metrics": metrics,
    }

    torch.save(checkpoint, config.LSTM_MODEL_PATH)
    print(f"Saved LSTM model to {config.LSTM_MODEL_PATH}")
    print(
        "Validation metrics: "
        f"MSE={metrics['mse']:.3f}, RMSE={metrics['rmse']:.3f}, "
        f"MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}"
    )


if __name__ == "__main__":
    main()
