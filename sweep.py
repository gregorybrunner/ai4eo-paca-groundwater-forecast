"""Grid-sweep (time_steps, forecast_steps) x BLOCKS: run RF, LSTM, and Ti-Rex2
against the persistence and seasonal-climatology baselines for each config
within each time block (its own chronological 80/20 split), report per-lead-day
+ total MSE, plot per block, and summarize mean +/- std across blocks.

Run:
    Edit EPOCHS/PLOT_PATH/CSV_PATH/GRID/BLOCKS below, then:
    python sweep.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baseline import persistence_mse, seasonal_climatology_mse
from data import load_dataset
from train import eval_tirex2, train_lstm, train_rf

GRID = [(180,90), (360, 90), (360, 180)]
EPOCHS = 60


BLOCKS = [
    ("2004-10-01", "2008-09-30"),
    ("2008-10-01", "2012-09-30"),
    ("2012-10-01", "2016-09-30"),
    ("2016-10-01", "2020-09-30"),
    ("2020-10-01", "2024-09-30"),
]


COLORS = {
    "persistence": "#767676",
    "climatology": "#B0B0B0",
    "rf": "#009E73",
    "lstm": "#0072B2",
    "tirex2": "#E69F00",
}
STYLES = {
    "persistence": "--",
    "climatology": ":",
    "rf": "-",
    "lstm": "-",
    "tirex2": "-",
}

BASELINE_FNS = {"persistence": persistence_mse, "climatology": seasonal_climatology_mse}

LABELS = {
    "persistence": "persistence",
    "climatology": "seasonal_climatology",
    "rf": "RF",
    "lstm": "LSTM",
    "tirex2": "Ti-Rex2",
}

RUN_NAME = "final-modis"
PLOT_PATH = f"results/sweep_{RUN_NAME}.png"
CSV_PATH = f"results/sweep_{RUN_NAME}.csv"


def plot_grid(results, path):
    """results: list of (time_steps, forecast_steps, {model_name: day_mse})."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, (time_steps, forecast_steps, curves) in zip(axes, results):
        lead_days = np.arange(1, forecast_steps + 1)
        for name in ("persistence", "climatology", "rf", "lstm", "tirex2"):
            ax.plot(lead_days, curves[name], label=LABELS[name], color=COLORS[name],
                    linestyle=STYLES[name], linewidth=2)
        ax.set_title(f"lookback={time_steps}d, horizon={forecast_steps}d", fontsize=10)
        ax.set_xlabel("lead day")
        ax.set_ylabel("MSE")

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


def main():
    Path(PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    print(f"{'block':>21}  {'lookback':>8}  {'horizon':>7}  {'model':>11}  "
          f"{'total_mse':>9}  {'last_day_mse':>12}")

    # (time_steps, forecast_steps, model_name) -> list of total_mse, one per block
    summary = {}
    rows = []  # one row per (block, config, model), written to CSV_PATH at the end

    for date_start, date_end in BLOCKS:
        block_results = []
        for time_steps, forecast_steps in GRID:
            ds = load_dataset(time_steps=time_steps, forecast_steps=forecast_steps,
                               date_start=date_start, date_end=date_end)
            if len(ds.X_test) == 0 or len(ds.X_train) == 0:
                print(f"{date_start}..{date_end}  lookback={time_steps}d horizon={forecast_steps}d  "
                      f"skipped: block too short for this config's train/test split")
                continue
            
            persistence_day_mse = BASELINE_FNS["persistence"](
                time_steps=time_steps, forecast_steps=forecast_steps,
                date_start=date_start, date_end=date_end)
            climatology_day_mse = BASELINE_FNS["climatology"](
                time_steps=time_steps, forecast_steps=forecast_steps,
                date_start=date_start, date_end=date_end)
            _, rf_day_mse = train_rf(ds)
            _, lstm_day_mse = train_lstm(ds, epochs=EPOCHS)
            _, tirex2_day_mse = eval_tirex2(ds)
            curves = {
                "persistence": persistence_day_mse, "climatology": climatology_day_mse,
                "rf": rf_day_mse, "lstm": lstm_day_mse, "tirex2": tirex2_day_mse,
            }

            for name, day_mse in curves.items():
                label = LABELS[name]
                print(f"{date_start}..{date_end}  {time_steps:>7}d  {forecast_steps:>6}d  "
                      f"{label:>11}  {day_mse.mean():>9.4f}  {day_mse[-1]:>12.4f}")
                summary.setdefault((time_steps, forecast_steps, label), []).append(day_mse.mean())
                rows.append({
                    "block_start": date_start, "block_end": date_end,
                    "time_steps": time_steps, "forecast_steps": forecast_steps,
                    "model": label, "total_mse": day_mse.mean(), "last_day_mse": day_mse[-1],
                    "day_mse": ";".join(f"{v:.6f}" for v in day_mse),
                })

            block_results.append((time_steps, forecast_steps, curves))

        plot_grid(block_results, f"{PLOT_PATH.removesuffix('.png')}_{date_start}_{date_end}.png")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {CSV_PATH}")

    print(f"\n=== summary across {len(BLOCKS)} blocks (mean +/- std of total MSE) ===")
    for (time_steps, forecast_steps, label), values in summary.items():
        arr = np.array(values)
        print(f"lookback={time_steps:>3}d horizon={forecast_steps:>2}d  {label:>11}: "
              f"{arr.mean():.4f} +/- {arr.std():.4f}  (n={len(arr)} blocks)")


if __name__ == "__main__":
    main()
