"""MODIS vegetation indices (NDVI/EVI).

Collection: MODIS/061/MOD13A2 (Terra, 1 km, 16-day composites, 2000-02-18+).

Zone comes from era5.py, so there is one definition of the PACA well footprint.

Daily series: linear interpolation between composites.

Output:  data/modis/modis_vi_daily.csv

Run:
  export EARTHENGINE_PROJECT=<your-gcp-project>
  python modis.py
"""
import datetime as dt
import os
from pathlib import Path

import ee
import pandas as pd

from era5 import (DAY_START, RANGE_END, build_well_zone, getinfo_retry,
                  load_well_points, to_ms)

COLLECTION = "MODIS/061/MOD13A2"
MODIS_SCALE_M = 1000
COMPOSITE_DAYS = 16
CHUNK_DAYS = 365

BANDS = {"NDVI": "ndvi", "EVI": "evi"}
QA_BAND = "SummaryQA"

QA_MAX = 1
VI_SCALE = 0.0001

OUT_DIR = Path("data/modis")
OUT_CSV = OUT_DIR / "modis_vi_daily.csv"


def _masked(img: ee.Image) -> ee.Image:
    """Drop snow/ice and cloud pixels, keep only the VI bands."""
    return img.updateMask(img.select(QA_BAND).lte(QA_MAX)).select(list(BANDS))


def reduce_chunk(start: dt.date, end: dt.date, zones: ee.FeatureCollection) -> list[dict]:
    """Zonal means per composite in [start, end), one request.

    Same toBands() as era5.reduce_chunk, but match on the band suffix.
    """

    coll = ee.ImageCollection(COLLECTION) \
        .filterDate(ee.Date(to_ms(start)), ee.Date(to_ms(end)))
    if coll.size().getInfo() == 0:      # short tail chunk past the archive end
        print(f"      no composites in {start}..{end}, skipping", flush=True)
        return []
    img = coll.map(_masked).toBands()
    stats = img.reduceRegions(collection=zones, reducer=ee.Reducer.mean(),
                              scale=MODIS_SCALE_M)

    rows = {}
    for props in getinfo_retry(stats):
        zid = props.get("zone_id")

        for key, val in props.items():
            for band, col in BANDS.items():
                if key.endswith(f"_{band}"):
                    datestr = key[: -(len(band) + 1)]
                    row_key = (zid, datestr)

                    if row_key not in rows:
                        rows[row_key] = {
                            "zone_id": zid,
                            "date": datestr
                        }

                    rows[row_key][col] = val
    return list(rows.values())


def pull_raw(zones: ee.FeatureCollection) -> pd.DataFrame:
    rows, d = [], DAY_START
    while d < RANGE_END:
        e = min(d + dt.timedelta(days=CHUNK_DAYS), RANGE_END)
        print(f"    {d} .. {e - dt.timedelta(days=1)}", flush=True)
        rows.extend(reduce_chunk(d, e, zones))
        d = e
    return pd.DataFrame(rows)


def convert_units(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the 0.0001 scale factor; stamp composites to their END date."""
    df = raw.copy()
    df["date"] = (pd.to_datetime(df["date"], format="%Y_%m_%d")
                  + pd.Timedelta(days=COMPOSITE_DAYS - 1))
    for col in BANDS.values():
        df[col] = df[col] * VI_SCALE
    return df.sort_values("date").reset_index(drop=True)


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """16-day composites, one row per day, interpolated between observations."""
    cols = list(BANDS.values())
    daily = (df.groupby("date")[cols].mean()
               .reindex(pd.date_range(DAY_START, RANGE_END - dt.timedelta(days=1),
                                      freq="D"))
               .interpolate(method="time", limit_area="inside"))
    return daily.rename_axis("date").reset_index()


def main() -> None:
    project = os.environ.get("EARTHENGINE_PROJECT")
    ee.Initialize(project=project) if project else ee.Initialize()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points = load_well_points()
    print(f"zone 'wells' ({', '.join(BANDS.values())}): {len(points)} wells")

    daily = to_daily(convert_units(pull_raw(build_well_zone(points))))
    daily.to_csv(OUT_CSV, index=False)

    obs = daily["ndvi"].notna().sum()

    print(f"\nWrote {OUT_CSV}: {len(daily)} days, "
          f"{daily.date.min().date()}..{daily.date.max().date()}")
    print(f"days with ndvi (after QA mask + interpolation): {obs} / {len(daily)}")
    print(daily[["ndvi", "evi"]].describe().to_string())


if __name__ == "__main__":
    main()
