"""Model classes for:

    - LSTM: Simple torch LSTM linear prediction head for the N forecast steps.
    Also helper Dataset class for LSTM.
    - Ti-Rex2: Pretrained TS foundation model (https://github.com/NX-AI/tirex-2)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from tirex2 import TimeseriesType, load_model


# LSTM Dataset
class TimeSeriesDataset(TorchDataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM Model
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_steps=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(in_features=hidden_size, out_features=forecast_steps)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

# Ti-Rex2 Model
class TiRex2Forecast:
    def __init__(self, path="NX-AI/TiRex-2", device="cpu"):
        self.model = load_model(path, device=device)

    def forecast(self, X, X_raw, target_idx, covariate_idx, prediction_length):
        """X: scaled windows (n, time_steps, n_features), used for covariates.
        X_raw: unscaled windows (same shape), used for the target, that must match
        the raw scale of y_test (see data.py's Dataset.X_*_raw).
        Return: the median-quantile forecast, shape (n, prediction_length).
        """
        series_list = [
            TimeseriesType(
                target=torch.tensor(w_raw[:, target_idx], dtype=torch.float32).unsqueeze(0),
                past_covariates=torch.tensor(w[:, covariate_idx].T, dtype=torch.float32),
                future_covariates=None)
            for w, w_raw in zip(X, X_raw)
        ]
        forecasts = self.model.forecast(series_list, prediction_length=prediction_length,
                                         output_type="numpy")
        return np.stack([f[0, 4, :] for f in forecasts])
