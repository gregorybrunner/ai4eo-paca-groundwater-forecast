# Predicting Seasonal Water Shortage in Provence-Alpes-Côte d'Azur from Earth Observation

This project forecasts groundwater levels for the **Provence-Alpes-Côte d'Azur (PACA)**
region of France.

The target is the **regional standardized groundwater level** `gwl_z`, which is 
a daily mean over ~175 piezometers collected by ADES, the official public portal for groundwater data.

Given a **lookback window** of the last *L* days of all features, 
the models forecast the next *H* days of `gwl_z`. 
Our forecast periods span more than 20 years, from **01.04.2002 - 02.10.2024**.

We compare three models:

- **Random Forest**: Classical ML approach.
- **Long Short-Term Memory** (LSTM) recurrent neural network.
- **Ti-Rex 2**: Zero-shot time-series foundation model.

They are scored against two baselines, persistence and seasonal
climatology, on identical windows, across several disjoint multi-year time
blocks.
**Persistence baseline** predicts the last lookback day value for all future values.
**Seasonal climatology baseline** predicts the historical / train-data mean of the day of the year.  

---

## Data sources

| Source | Link | License |
|--------|------|---------|
| **ADES** | [Hub'Eau API](https://hubeau.eaufrance.fr/page/api-piezometrie) | Etalab Open Licence 2.0 |
| **ERA5-Land** | [GEE: ECMWF/ERA5_LAND/DAILY_AGGR](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR) | Copernicus licence |
| **GRACE / GRACE-FO** | [GEE: NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI](https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_V04_MASCON_CRI) | Open (NASA/JPL) |
| **MODIS** | [GEE: MODIS/061/MOD13A2](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A2) | Open (NASA/USGS, LP DAAC) |

---

## Pipeline

Four acquisition scripts each write one daily CSV under `data/`:
`groundwater.py` pulls the per-well panel from Hub'Eau, `era5.py`, `grace.py`
and `modis.py` pull their variables from Google Earth Engine. `groundwater.py` runs
first, because the well coordinates it writes define the `wells` zone used by both
`era5.py` and `modis.py`. `build_dataset.py` then merges the four into the
single model-ready table `data/dataset_paca_daily.csv`.

From there, `data.py` is the single place where windows, scaling and the
train/test split are built, so every model is scored on exactly the same
windows. `baseline.py`, `train.py` and `sweep.py` all consume it.

---

## Files

| File | What it does |
|------|--------------|
| `regions.py` | Department code → name maps for PACA (04/05/06/13/83/84) and Occitanie. |
| `groundwater.py` | Pulls every PACA piezometer from Hub'Eau, keeps wells with ≥500 measurements and ≥3 years of record, puts each on a daily grid (linear interpolation across gaps ≤31 days), z-scores each well on its own record → long panel `date, code_bss, dep, lon, lat, gwl_raw, gwl_z, interpolated`. |
| `era5.py` | Earth Engine → daily zonal means of the ERA5-Land bands, per zone (`alps` / `wells`), chunked 90 days per request. |
| `grace.py` | Earth Engine → monthly GRACE mascon mean over the PACA departments, broadcast to daily; writes a `raw` version (gap months NaN) and a `filled` version (linearly interpolated). |
| `modis.py` | Earth Engine → NDVI/EVI zonal means over the `wells` footprint from 16-day MODIS composites, linearly interpolated to a daily series. |
| `build_dataset.py` | Merges ERA5 + GRACE + MODIS + the per-well panel (averaged to one regional `gwl_z` per day) → `data/dataset_paca_daily.csv`. |
| `data.py` | Loads the dataset, adds `doy_sin`/`doy_cos`, scales features, builds lookback/horizon windows, applies the chronological 80/20 split, and computes the persistence anchor. |
| `models.py` | `LSTMForecast` (1-layer LSTM + linear head emitting all *H* steps), its torch `Dataset` wrapper, and `TiRex2Forecast` (wraps the pretrained NX-AI Ti-Rex 2 checkpoint). |
| `baseline.py` | Persistence and seasonal climatology functions. |
| `train.py` | Trains/evaluates RF, LSTM and Ti-Rex 2 on **one** `(lookback, horizon)` config and prints the comparison table vs. persistence. |
| `sweep.py` | Runs the whole comparison over a **grid of configs × time blocks**, writes per-lead-day curves to CSV, one plot per block, and a mean ± std summary across blocks. |

---

## How to run

### 1. Install

The project uses Python 3.12, PyTorch 2.9.1.

```bash
git clone https://github.com/gregorybrunner/ai4eo-paca-groundwater-forecast
cd ai4eo-paca-groundwater-forecast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Rebuild the dataset

**Order matters**: `era5.py` and `modis.py` read the well coordinates written by
`groundwater.py` to build the `wells` zone. `groundwater.py` takes ~15 minutes
(one API call per well, rate-limited).

```bash
earthengine authenticate                        # one-time
export EARTHENGINE_PROJECT=<your-gcp-project>

python groundwater.py     # Hub'Eau  -> data/gwl/daily_gwl_wells_paca.csv
python era5.py            # GEE      -> data/era5/era5_features_daily.csv
python grace.py           # GEE      -> data/grace/daily_{raw,filled}_GRACE_PACA.csv
python modis.py           # GEE      -> data/modis/modis_vi_daily.csv
python build_dataset.py   # merge    -> data/dataset_paca_daily.csv
```

### 3. Train and evaluate

```bash
python train.py        # RF + LSTM + Ti-Rex 2 on one config
python sweep.py        # full grid x blocks -> results/*.csv + *.png
```

`train.py` is configured by the constants at the top of the file:

```python
MODELS = ["rf", "lstm", "tirex2"]
TIME_STEPS = 180        # lookback, days
FORECAST_STEPS = 90     # horizon, days
EPOCHS = 10
SEED = 2026
```

`sweep.py` likewise:

```python
GRID   = [(180, 90), (360, 90), (360, 180)]   # (lookback, horizon)
EPOCHS = 60
BLOCKS = [("2004-10-01", "2008-09-30"), ..., ("2020-10-01", "2024-09-30")]
RUN_NAME = "run-xyz"              # names the output files
```

Each block gets its own chronological 80/20 split.
Outputs land in
`results/`:
`sweep_<RUN_NAME>.csv` (one row per block × config × model, with the full
per-lead-day MSE curve) and `sweep_<RUN_NAME>_<block>.png`.

The first Ti-Rex 2 run downloads the `NX-AI/TiRex-2` checkpoint from Hugging
Face and caches it.
