"""Download ADES piezometry (via Hub'Eau) and build a daily, per-well
groundwater panel for PACA.

Each well still gets its own standardization z-score.
z>0 = wetter than that well's normal, z<0 = drier.

Output (long panel): data/gwl/daily_gwl_wells_paca.csv
  date, code_bss, dep, lon, lat, gwl_raw, gwl_z, interpolated

Run: python groundwater.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from regions import PACA

BASE = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes"
START = "2000-01-01" # ~ MODIS, GRACE start date            
END = "2026-06-01"     

DEPARTEMENTS = list(PACA)   

MIN_OBS = 500       # min nr of measurements to trust a well's baseline
MIN_YEARS = 3.0     # in record span
MAX_GAP_DAYS = 31   # interpolate daily gaps up to this length

LEVEL_FIELD = "niveau_nappe_eau"   # NGF water-table elevation
OUT_DIR = Path("data/gwl")


def _get_all(url, params):
    """GET a Hub'Eau endpoint, following the next cursor."""
    rows: list[dict] = []

    # next URL carries the next query
    while url:
        r = requests.get(url, params=params, timeout=120)
        if r.status_code not in (200, 206):
            r.raise_for_status()
        j = r.json()
        rows += j.get("data", [])
        url = j.get("next")
        params = None         
    return rows


def resolve_stations():
    """Fetch every PACA-department well + its metadata (code, dep, lon/lat,
    record length), then keep only wells with enough history to trust their
    per-well z-score baseline."""
    meta = pd.DataFrame(_get_all(
        f"{BASE}/stations",
        {"code_departement": ",".join(DEPARTEMENTS), "format": "json", "size": 5000},
    ))
    meta = meta.rename(columns={"code_departement": "dep", "x": "lon", "y": "lat"})
    meta = meta[["code_bss", "dep", "lon", "lat",
                 "date_fin_mesure", "nb_mesures_piezo"]].copy()
    print(f"PACA departments {DEPARTEMENTS}: {len(meta)} wells")

    nb = pd.to_numeric(meta["nb_mesures_piezo"], errors="coerce").fillna(0)
    fin = pd.to_datetime(meta["date_fin_mesure"], errors="coerce")
    meta = meta[(nb >= MIN_OBS) & (fin >= pd.Timestamp(START))].reset_index(drop=True) # filter
    print(f"after pre-filter (>= {MIN_OBS} obs, active >= {START}): {len(meta)} wells")
    return meta


def fetch_well_daily(code_bss):
    """All measurements for one well from START to a DAILY-mean Series."""
    data = _get_all(f"{BASE}/chroniques",
                    {"code_bss": code_bss, "date_debut_mesure": START,
                     "size": 20000})
    if not data:
        return None
    df = pd.DataFrame(data)
    if LEVEL_FIELD not in df or df[LEVEL_FIELD].isna().all():
        return None
    df["date"] = pd.to_datetime(df["date_mesure"])
    df = df.dropna(subset=[LEVEL_FIELD])
    daily = df.groupby(df["date"].dt.floor("D"))[LEVEL_FIELD].mean()
    return daily.sort_index()


def to_daily_grid(raw):
    """Reindex a well to a full daily grid, interpolate short gaps.

    Returns columns: gwl_raw, interpolated (bool). Days beyond MAX_GAP_DAYS from
    any real reading stay NaN and are dropped.
    """
    grid = pd.date_range(raw.index.min(), min(raw.index.max(), pd.Timestamp(END)),
                         freq="D")
    s = raw.reindex(grid)
    is_real = s.notna()
    # linearly interpolate, but only across gaps <= MAX_GAP_DAYS
    filled = s.interpolate(method="time", limit=MAX_GAP_DAYS, limit_area="inside")
    out = pd.DataFrame({"gwl_raw": filled, "interpolated": ~is_real & filled.notna()})
    return out.dropna(subset=["gwl_raw"])


def standardize(gwl_raw):
    """Per-well z-score."""
    base = gwl_raw
    mu, std = base.mean(), base.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(np.nan, index=gwl_raw.index)
    return (gwl_raw - mu) / std


def main():
    meta = resolve_stations()

    min_days = MIN_YEARS * 365.25
    frames: list[pd.DataFrame] = []
    n_ok = 0
    for i, row in enumerate(meta.itertuples(index=False), 1):
        code = str(row.code_bss)
        try:
            raw = fetch_well_daily(code)
        except Exception as e:                       
            print(f"    [{i}/{len(meta)}] {code}: error {str(e)[:60]}")
            raw = None
        if raw is None:
            continue

        grid = to_daily_grid(raw)
        span_days = (grid.index.max() - grid.index.min()).days
        if len(grid) < min_days or span_days < min_days:
            continue
        grid["gwl_z"] = standardize(grid["gwl_raw"])
        grid = grid.dropna(subset=["gwl_z"])
        grid = grid.reset_index().rename(columns={"index": "date"})
        grid["code_bss"] = code
        grid["dep"] = getattr(row, "dep", None)
        grid["lon"] = getattr(row, "lon", None)
        grid["lat"] = getattr(row, "lat", None)
        frames.append(grid)
        n_ok += 1
        if i % 20 == 0:
            print(f"{i}/{len(meta)} wells scanned, {n_ok} kept", flush=True)
        time.sleep(0.15) # API is slow...

    if not frames:
        raise SystemExit("No usable wells.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[["date", "code_bss", "dep", "lon", "lat",
                   "gwl_raw", "gwl_z", "interpolated"]]
    panel = panel.sort_values(["code_bss", "date"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_DIR / "daily_gwl_wells_paca.csv", index=False)


    n_wells = panel["code_bss"].nunique()
    interp_share = panel["interpolated"].mean()
    
    print(f"\nWrote {len(panel):,} well-days from {n_wells} wells to "
          f"data/daily_gwl_wells_paca.csv")
    print(f"span {panel.date.min().date()} .. {panel.date.max().date()}")
    print(f"interpolated share: {interp_share:.1%}  "
          f"(rest are real daily means)")
    print(f"median days/well: {int(panel.groupby('code_bss').size().median()):,}")


if __name__ == "__main__":
    main()
