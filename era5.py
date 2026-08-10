"""Daily ERA5-Land soil/meteo features for the PACA water-shortage model.

ERA5-Land daily aggregates on Earth Engine
(ECMWF/ERA5_LAND/DAILY_AGGR, GEE mirror of the Copernicus CDS ERA5-Land
product, ~9 km). 

To pull another variable, add ONE entry to FEATURES below.

Each feature is extracted over the ZONE that physically drives it: wells or alps.

The 'wells' zone is built from the well coordinates in WELLS_CSV: every well is
buffered by WELL_BUFFER_M (~ one ERA5-Land cell) and the buffers are dissolved
into a single PACA-wide footprint, so the forcing is averaged over the same
wells that feed the gwl_z target in build_dataset.py.

Output:
  data/era5/era5_features_daily.csv

Run:
  export EARTHENGINE_PROJECT=<your-gcp-project>
  python era5.py
"""
import datetime as dt
import os
import time
from pathlib import Path

import ee
import pandas as pd

RANGE_END = dt.date(2026, 1, 1)     # exclusive

# Bounding box coordinates of the areas of interest
AOIS = {
    "durance_upper": [6.20, 44.35, 6.95, 44.95],   # Serre-Poncon / Ubaye / Embrunais
    "verdon_upper":  [6.30, 43.85, 6.95, 44.30],   # upper Verdon -> Sainte-Croix
    "var_haut":      [6.70, 43.95, 7.45, 44.40],   # Haut-Var / Mercantour
}

WELLS_CSV = Path("data/gwl/daily_gwl_wells_paca.csv")
WELL_BUFFER_M = 10_000


def build_basins(aois: dict[str, list[float]] = AOIS) -> ee.FeatureCollection:
    """Create rectangular features for the fixed upstream snow basins in the Alps zone."""

    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Rectangle(box), {"zone_id": bid})
        for bid, box in aois.items()
    ])


def load_well_points(path: Path = WELLS_CSV) -> list[list[float]]:
    """Load unique well coordinates from CSV file"""

    wells = pd.read_csv(path, usecols=["code_bss", "lon", "lat"])
    wells = wells.drop_duplicates("code_bss").dropna(subset=["lon", "lat"])
    return wells[["lon", "lat"]].to_numpy().tolist()


def build_well_zone(points: list[list[float]]) -> ee.FeatureCollection:
    """Build a PACA-wide well zone from buffered well locations."""

    zone = ee.Geometry.MultiPoint(points).buffer(WELL_BUFFER_M)
    return ee.FeatureCollection([ee.Feature(zone, {"zone_id": "paca"})])


def to_ms(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day,
                           tzinfo=dt.timezone.utc).timestamp() * 1000)


def getinfo_retry(fc: ee.FeatureCollection) -> list[dict]:
    """Get feature properties, retrying on temporary EE limits."""

    for attempt in range(6):
        try:
            return [f["properties"] for f in fc.getInfo()["features"]]
        except ee.ee_exception.EEException as e:
            if "concurrent" in str(e).lower() or "429" in str(e):
                wait = 5 * (2 ** attempt)
                print(f"      429/limit, retry in {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("getinfo_retry: exhausted retries on EE rate limit")


# ERA5-Land features to extract
FEATURES = {
    "swe_mm": {
        "band": "snow_depth_water_equivalent",
        "scale": 1000.0, "offset": 0.0, "zone": "alps",
    },
    "soil_temp_c": {
        "band": "soil_temperature_level_1",
        "scale": 1.0, "offset": -273.15, "zone": "wells",
    },
    "soil_moisture": {
        "band": "volumetric_soil_water_layer_1",
        "scale": 1.0, "offset": 0.0, "zone": "wells",
    },
    "precip_mm": {
        "band": "total_precipitation_sum",
        "scale": 1000.0, "offset": 0.0, "zone": "wells",
    },
    "soil_moisture_l4": {
        "band": "volumetric_soil_water_layer_4",
        "scale": 1.0, "offset": 0.0, "zone": "wells",
    },
    "sub_surface_runoff_mm": {
        "band": "sub_surface_runoff_sum",
        "scale": 1000.0, "offset": 0.0, "zone": "wells",
    },
}


def zone_features(zone: str) -> dict[str, dict]:
    """Select the features for a specific zone"""
    return {n: s for n, s in FEATURES.items() if s["zone"] == zone}


def band_to_name(features: dict[str, dict]) -> dict[str, str]:
    return {spec["band"]: name for name, spec in features.items()}


DAY_START = dt.date(2000, 1, 1)
ERA5_SCALE_M = 9000
CHUNK_DAYS = 90
COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"

OUT_DIR = Path("data/era5")



# Earth Engine
def reduce_chunk(start: dt.date, end: dt.date, zones: ee.FeatureCollection,
                 features: dict[str, dict]) -> list[dict]:
    """Compute daily zonal means for a date range in a single request."""

    names = band_to_name(features)
    img = ee.ImageCollection(COLLECTION) \
        .filterDate(ee.Date(to_ms(start)), ee.Date(to_ms(end))) \
        .select(list(names)).toBands()

    stats = img.reduceRegions(collection=zones,
                              reducer=ee.Reducer.mean(), scale=ERA5_SCALE_M)

    rows = {}
    for props in getinfo_retry(stats):
        zid = props.get("zone_id")

        for key, val in props.items():
            if "_" not in key:
                continue

            datestr, band = key.split("_", 1)
            name = names.get(band)

            if name is None or len(datestr) != 8 or not datestr.isdigit():
                continue

            row_key = (zid, datestr)

            if row_key not in rows:
                rows[row_key] = {
                    "zone_id": zid,
                    "date": datestr
                }

            rows[row_key][name] = val

    return list(rows.values())


def pull_raw(zones: ee.FeatureCollection, features: dict[str, dict]) -> pd.DataFrame:
    """Chunked pull over the full date range, one EE request per chunk."""
    chunk_starts = []
    d = DAY_START
    while d < RANGE_END:
        chunk_starts.append(d)
        d += dt.timedelta(days=CHUNK_DAYS)

    print(f"  {DAY_START} .. {RANGE_END}, {len(chunk_starts)} chunks of "
          f"{CHUNK_DAYS} days ...", flush=True)

    rows = []
    for s in chunk_starts:
        e = min(s + dt.timedelta(days=CHUNK_DAYS), RANGE_END)
        print(f"    days {s} .. {e - dt.timedelta(days=1)}", flush=True)

        rows.extend(reduce_chunk(s, e, zones, features))

    return pd.DataFrame(rows)


# Post-processing
def convert_units(raw: pd.DataFrame, features: dict[str, dict]) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    for name, spec in features.items():
        df[name] = df[name] * spec["scale"] + spec["offset"]

    return df


def to_daily(df: pd.DataFrame, features: dict[str, dict]) -> pd.DataFrame:
    """Collapse the per-zone panel to one row per day.

    Plain mean across zones: the alpine basins are equal-standing catchments,
    and the wells zone is a single PACA-wide footprint anyway.
    """
    daily = df.groupby("date")[list(features)].mean()
    return daily.reset_index().sort_values("date").reset_index(drop=True)


def main() -> None:
    project = os.environ.get("EARTHENGINE_PROJECT")

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    alps_features = zone_features("alps")
    well_features = zone_features("wells")

    print(f"zone 'alps'  ({', '.join(alps_features)}): {len(AOIS)} basins")
    raw_alps = pull_raw(build_basins(), alps_features)
    daily_alps = to_daily(convert_units(raw_alps, alps_features), alps_features)

    points = load_well_points()
    print(f"zone 'wells' ({', '.join(well_features)}): {len(points)} wells, "
          f"{WELL_BUFFER_M / 1000:.0f} km buffer")

    raw_wells = pull_raw(build_well_zone(points), well_features)
    daily_wells = to_daily(convert_units(raw_wells, well_features), well_features)

    daily = daily_alps.merge(daily_wells, on="date", how="outer")
    daily = daily[["date"] + list(FEATURES)].sort_values("date").reset_index(drop=True)
    daily.to_csv(OUT_DIR / "era5_features_daily.csv", index=False)

    print(f"\nDAILY: {len(daily)} days, "
          f"{daily.date.min().date()}..{daily.date.max().date()}")
    print("\nDAILY head:")
    print(daily.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
