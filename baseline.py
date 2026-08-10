"""Baseline forecasts for groundwater level (gwl_z) prediction.

Chronological 80/20 train/test split, with lookback days and horizon days.

Two baselines:
    - Persistence: Simply predict the last day of the training data.
    - Seasonal Climatology: Predict the historical mean for each day.

Run:  python baseline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import DATE_END, DATE_START, split_starts

DATA_PATH = Path("data/dataset_paca_daily.csv")
TARGET_COL = "gwl_z"
TRAIN_FRAC = 0.8


def _load_series(data_path, date_start=DATE_START, date_end=DATE_END, target_col=TARGET_COL):
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= date_start) & (df["date"] <= date_end)]
    return df[["date", target_col]].reset_index(drop=True)


def _test_split(df, target_col, time_steps, forecast_steps, train_frac):
    """Test-set forecast targets: (y, train_size, test_starts, target_idx, y_true).

    test_starts are lookback-window start indices held out (cutoff at train_size on the start
    index).
    """
    y = df[target_col].to_numpy()
    if len(y) - time_steps - forecast_steps + 1 < 1:
        raise ValueError("time_steps + forecast_steps exceeds the series length")

    train_size = int(len(y) * train_frac)
    _, test_starts = split_starts(len(y), train_size, time_steps, forecast_steps)
    target_idx = test_starts[:, None] + time_steps + np.arange(forecast_steps)[None, :]
    y_true = y[target_idx]
    return y, train_size, test_starts, target_idx, y_true


def persistence_mse(time_steps=180, forecast_steps=30, train_frac=TRAIN_FRAC,
                     data_path=DATA_PATH, date_start=DATE_START, date_end=DATE_END, target_col=TARGET_COL):
    """Naive persistence baseline: hold the last known value flat over the whole horizon.

    Returns per-lead-day MSE, shape (forecast_steps,)
    """
    df = _load_series(data_path, date_start, date_end, target_col)
    y, train_size, test_starts, target_idx, y_true = _test_split(
        df, target_col, time_steps, forecast_steps, train_frac)

    last_known = y[test_starts + time_steps - 1]
    y_pred = np.repeat(last_known[:, None], forecast_steps, axis=1)
    return np.mean((y_true - y_pred) ** 2, axis=0)


def seasonal_climatology_mse(time_steps=180, forecast_steps=30, train_frac=TRAIN_FRAC,
                             data_path=DATA_PATH, date_start=DATE_START, date_end=DATE_END, target_col=TARGET_COL):
    """Climatology baseline: predict each date's train-only historical mean
    for that calendar day (month-day), independent of lookback/horizon.

    Returns per-lead-day MSE, shape (forecast_steps,)
    """
    df = _load_series(data_path, date_start, date_end, target_col)
    y, train_size, test_starts, target_idx, y_true = _test_split(
        df, target_col, time_steps, forecast_steps, train_frac)

    month_day = df["date"].dt.strftime("%m-%d").to_numpy()
    climatology = pd.Series(y[:train_size], index=month_day[:train_size]).groupby(level=0).mean()
    fallback = y[:train_size].mean()  # covers a month-day never seen in training (leap day)

    y_pred_by_date = np.array([climatology.get(d, fallback) for d in month_day])
    y_pred = y_pred_by_date[target_idx]
    return np.mean((y_true - y_pred) ** 2, axis=0)
