"""Assemble the full daily data csv.

Contains: daily ERA5 extraction, MODIS vegetation indices, GRACE groundwater,
target ground truth gwl well levels:
    - data/era5/era5_features_daily.csv
    - data/modis/modis_vi_daily.csv
    - data/grace/daily_filled_GRACE_PACA.csv
    - data/gwl/daily_gwl_wells_paca.csv

Output: data/dataset_paca_daily.csv

Run: python build_dataset_daily.py
"""
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data")
REGION = "paca"

ERA5_DAILY = "data/era5/era5_features_daily.csv"
MODIS_DAILY = "data/modis/modis_vi_daily.csv"
GRACE_DAILY = f"data/grace/daily_filled_GRACE_{REGION.upper()}.csv"
GWL_DAILY = "data/gwl/daily_gwl_wells_paca.csv"


def _grace_daily(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df[["date", "GRACE_LWE_Anomaly_cm"]]


def _gwl_daily(path):
    """Per-well gwl_z -> one regional value per day (mean across wells)."""
    wells = pd.read_csv(path, usecols=["date", "gwl_z"], parse_dates=["date"])
    return wells.groupby("date")["gwl_z"].mean().reset_index()


def build():
    # daily features
    df = pd.read_csv(ERA5_DAILY, parse_dates=["date"])

    df = df.merge(pd.read_csv(MODIS_DAILY, parse_dates=["date"]), on="date", how="left")
    df = df.merge(_grace_daily(GRACE_DAILY), on="date", how="left")
    df = df.merge(_gwl_daily(GWL_DAILY), on="date", how="left")

    cols = ["date", "swe_mm", "soil_temp_c", "soil_moisture", "precip_mm",
            "soil_moisture_l4", "sub_surface_runoff_mm", "ndvi", "evi",
            "GRACE_LWE_Anomaly_cm", "gwl_z"]
    return df[cols].sort_values("date").reset_index(drop=True)


def main():
    df = build()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"dataset_{REGION}_daily.csv", index=False)

    labelled = df["gwl_z"].notna().sum()
    print(f"Built dataset_{REGION}_daily: {len(df)} days, "
          f"{df.date.min().date()} .. {df.date.max().date()}")
    print("columns:", list(df.columns))
    print(f"days with gt (gwl_z) label: {labelled} / {len(df)}")
    print(f"days with GRACE: {df['GRACE_LWE_Anomaly_cm'].notna().sum()} / {len(df)}")
    print()
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
