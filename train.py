"""Training loop of all three models on a given forecast and lookback period compared to baselines.

All models get the exact same windows from data.py: a TIME_STEPS-day history
of all features in, the next FORECAST_STEPS days of gwl_z out.

Run: Edit MODELS/TIME_STEPS/FORECAST_STEPS/EPOCHS, then: python train.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import LSTMForecast, TimeSeriesDataset, TiRex2Forecast

from baseline import persistence_mse
from data import DATE_END, DATE_START, load_dataset

MODELS = ["rf", "lstm", "tirex2"]
# MODELS = ["lstm"]
TIME_STEPS = 180
FORECAST_STEPS = 90
EPOCHS = 10
SEED = 2026

LSTM_DEVICE = "mps" if torch.mps.is_available() else "cpu"
TIREX2_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_lstm(model, loader, device):
    """Per-lead-day MSE, shape (forecast_steps,). Averaged across eval batches."""
    model.eval()
    batch_mses = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            y_batch = y_batch.squeeze(-1)
            batch_mses.append(((output - y_batch) ** 2).mean(dim=0))
    return torch.stack(batch_mses).mean(dim=0).cpu().numpy()


def train_lstm(ds, epochs=60, hidden_size=32, lr=3e-4, batch_size=32, device=LSTM_DEVICE, seed=SEED,
               val_frac=0.2, patience=5):
    torch.manual_seed(seed)
    device = torch.device(device)

    
    y_train = (ds.y_train.squeeze(-1) - ds.anchor_train)[..., None]
    y_test = (ds.y_test.squeeze(-1) - ds.anchor_test)[..., None]

    n_val = int(len(ds.X_train) * val_frac)
    n_inner = len(ds.X_train) - n_val - y_train.shape[1]

    train_loader = DataLoader(TimeSeriesDataset(ds.X_train[:n_inner], y_train[:n_inner]),
                               batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TimeSeriesDataset(ds.X_train[-n_val:], y_train[-n_val:]),
                             batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(TimeSeriesDataset(ds.X_test, y_test),
                              batch_size=batch_size, shuffle=False)

    model = LSTMForecast(input_size=len(ds.features), hidden_size=hidden_size,
                          forecast_steps=ds.y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val, best_state, best_epoch, stale = np.inf, None, 0, 0
    for e in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = eval_lstm(model, val_loader, device).mean()
        print(f"Epoch {e:>2}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, e, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    day_mse = eval_lstm(model, eval_loader, device)   # test touched once, at the end
    print(f"LSTM (best epoch {best_epoch}/{e}, val={best_val:.4f})  "
          f"eval_loss={day_mse.mean():.4f}  eval_last_day={day_mse[-1]:.4f}")
    return model, day_mse


def eval_tirex2(ds, device=TIREX2_DEVICE):
    forecaster = TiRex2Forecast(device=device)
    y_pred = forecaster.forecast(ds.X_test, ds.X_test_raw, ds.target_idx, ds.covariate_idx,
                                  prediction_length=ds.y_test.shape[1])
    y_true = ds.y_test.squeeze(-1)
    day_mse = np.mean((y_pred - y_true) ** 2, axis=0)

    print(f"Ti-Rex2 (zero-shot)  eval_loss={day_mse.mean():.4f}  eval_last_day={day_mse[-1]:.4f}")
    return forecaster, day_mse


def train_rf(ds, n_estimators=75, seed=SEED):
    """Random Forest: one flattened history window in -> all horizon days out.
    Predicts all forecast_steps days in one go.
    """
    from sklearn.ensemble import RandomForestRegressor

    X_train = ds.X_train.reshape(len(ds.X_train), -1)  # flatten the windows
    X_test = ds.X_test.reshape(len(ds.X_test), -1)

    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=25,
                                  n_jobs=-1, random_state=seed)
    model.fit(X_train, ds.y_train.squeeze(-1) - ds.anchor_train)

    pred = model.predict(X_test)
    day_mse = np.mean((pred - (ds.y_test.squeeze(-1) - ds.anchor_test)) ** 2, axis=0)

    print(f"RF ({n_estimators} trees)  eval_loss={day_mse.mean():.4f}  "
          f"eval_last_day={day_mse[-1]:.4f}")
    return model, day_mse


def main():
    ds = load_dataset(time_steps=TIME_STEPS, forecast_steps=FORECAST_STEPS)

    results = {}
    for name in MODELS:
        if name != "rf" and torch is None:
            print(f"skipping {name} (torch not installed)")
            continue
        if name == "rf":
            _, results[name] = train_rf(ds)
        elif name == "lstm":
            _, results[name] = train_lstm(ds, epochs=EPOCHS, hidden_size=12, device=LSTM_DEVICE)
        elif name == "tirex2":
            _, results[name] = eval_tirex2(ds, device=TIREX2_DEVICE)

    # reference every model against the naive "no change" forecast
    results["persistence"] = persistence_mse(
        time_steps=TIME_STEPS, forecast_steps=FORECAST_STEPS,
        date_start=DATE_START, date_end=DATE_END)

    p_mean = results["persistence"].mean()
    print(f"\n{'model':<12}{'mean MSE':>10}{'day 1':>9}"
          f"{'day ' + str(FORECAST_STEPS):>9}{'vs persistence':>16}")
    for name, day_mse in results.items():
        skill = 1 - day_mse.mean() / p_mean  # >0 = better than persistence
        print(f"{name:<12}{day_mse.mean():>10.4f}{day_mse[0]:>9.4f}"
              f"{day_mse[-1]:>9.4f}{skill:>15.0%}")


if __name__ == "__main__":
    main()
