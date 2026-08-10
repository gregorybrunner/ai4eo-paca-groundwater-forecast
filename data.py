"""Shared data preparation for every model (LSTM, RF, TiRex, baselines).

Loads dataset_paca_daily.csv, scales the feature set, builds lookback/horizon
windows, and applies the same chronological train/test split everywhere, so
all models are scored on identical windows and results are comparable across
lookback/horizon settings.

Usage:
    from data import load_dataset
    ds = load_dataset(time_steps=180, forecast_steps=30)
    ds.X_train, ds.y_train
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/dataset_paca_daily.csv")
FEATURES = ["swe_mm", "soil_temp_c", "soil_moisture", "GRACE_LWE_Anomaly_cm",
            "precip_mm", "ndvi", "evi", "doy_sin", "doy_cos", "gwl_z"]
TARGET_COL = "gwl_z"

# first GRACE month
DATE_START = "2002-04-01"
# last day with GRACE coverage
DATE_END = "2024-10-02"

TRAIN_FRAC = 0.8


class Dataset:
    def __init__(self, df, features, target_col, target_idx, covariate_idx, scaler: StandardScaler,
                 X_train, X_test, X_train_raw, X_test_raw, y_train, y_test,
                 anchor_train, anchor_test):
        self.df = df                          
        self.features = features
        self.target_col = target_col
        self.target_idx = target_idx
        self.covariate_idx = covariate_idx
        self.scaler = scaler
        self.X_train = X_train
        self.X_test = X_test
        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw
        self.y_train = y_train
        self.y_test = y_test
        self.anchor_train = anchor_train
        self.anchor_test = anchor_test


def _create_sequences(data, target, time_steps, forecast_steps):
    """create X,y pairs of lookback horizon, forecast period."""
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_steps + 1):
        X.append(data[i:i + time_steps])
        y.append(target[i + time_steps: i + time_steps + forecast_steps])
    return np.array(X), np.array(y)


def split_starts(n_rows, cutoff, time_steps, forecast_steps):
    """Window start indices, split by the days each window forecasts."""
    last = n_rows - time_steps - forecast_steps
    train = np.arange(0, min(cutoff - time_steps - forecast_steps, last) + 1)
    test = np.arange(max(cutoff - time_steps, 0), last + 1)
    return train, test


def load_dataset(time_steps, forecast_steps, train_frac=TRAIN_FRAC, features=FEATURES, target_col=TARGET_COL,
                  date_start=DATE_START, date_end=DATE_END, data_path=DATA_PATH):
    """
    Load + filter + window + split for a given lookback/horizon.
    """
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= date_start) & (df["date"] <= date_end)].reset_index(drop=True)

    doy = df["date"].dt.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    X = df[features].to_numpy().copy()
    y = df[[target_col]].to_numpy().copy()

    train_size = int(len(df) * train_frac)

    scaler = StandardScaler().fit(X[:train_size])
    X_scaled = scaler.transform(X)

    X_seq, y_seq = _create_sequences(X_scaled, y, time_steps, forecast_steps)
    X_seq_raw, _ = _create_sequences(X, y, time_steps, forecast_steps)

    tr, te = split_starts(len(df), train_size, time_steps, forecast_steps)
    target_idx = features.index(target_col)

    # Persistence anchor: each window's last observed day, held flat over the
    # horizon. Models fit y - anchor
    def anchor(starts):
        return np.repeat(y[starts + time_steps - 1], forecast_steps, axis=1)

    return Dataset(
        df=df, features=features, target_col=target_col,
        target_idx=target_idx,
        covariate_idx=[i for i in range(len(features)) if i != target_idx],
        scaler=scaler,
        X_train=X_seq[tr], X_test=X_seq[te],
        X_train_raw=X_seq_raw[tr], X_test_raw=X_seq_raw[te],
        y_train=y_seq[tr], y_test=y_seq[te],
        anchor_train=anchor(tr), anchor_test=anchor(te),
    )
