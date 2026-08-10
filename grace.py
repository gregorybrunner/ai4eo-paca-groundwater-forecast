"""Pull GRACE/GRACE-FO monthly mass grids from Earth Engine and build a daily
regional GRACE series for PACA region.

Values are kept in raw units (cm LWE anomaly, relative to the 2004-2009
baseline)

Output: data/grace/daily_raw_GRACE_PACA.csv,
data/grace/daily_filled_GRACE_PACA.csv
  date, GRACE_LWE_Anomaly_cm
  (raw: monthly solution broadcast to days, gap months NaN;
   filled: gap months linearly interpolated)

Run:
  export EARTHENGINE_PROJECT=<your-gcp-project>
  python grace.py
"""
import os
import ee
import pandas as pd
import datetime

from regions import PACA


def get_grace_for_region(dept_names, region_name, start_date='2002-04-01'):
    """
    Retrieve the monthly averaged GRACE water storage anomalies and
    assign each monthly value to each day of the corresponding month.

    Parameters:
        dept_names: the departments used to define the study region
        region_name: the study region
        start_date: start date for GRACE data

    Returns:
        final_df: a pandas dataframe
    """

    admin_regions = ee.FeatureCollection("FAO/GAUL/2015/level2") \
        .filter(ee.Filter.eq('ADM0_NAME', 'France')) \
        .filter(ee.Filter.inList('ADM2_NAME', dept_names))

    if admin_regions.size().getInfo() == 0:
        raise ValueError(f"No departments found for: {region_name}")

    region_geometry = admin_regions.geometry()

    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    grace_collection = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI") \
        .filterDate(start_date, end_date) \
        .select('lwe_thickness')

    def get_regional_mean(image):
        """
        Calculate the regional mean of a monthly GRACE water storage anomaly.

        Parameters:
            image: a monthly GRACE image

        Returns:
            An Earth Engine Feature containing the image date and regional mean
        """

        mean_dict = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region_geometry,
            scale=111320,
            maxPixels=1e9
        )
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'lwe_anomaly_cm': mean_dict.get('lwe_thickness')
        })

    regional_timeseries = grace_collection.map(get_regional_mean)
    grace_data = regional_timeseries.getInfo()['features']

    grace_rows = []

    for feature in grace_data:
        properties = feature['properties']
        row = {
            'date': properties['date'],
            'GRACE_LWE_Anomaly_cm': properties.get('lwe_anomaly_cm')
        }

        grace_rows.append(row)

    df_grace = pd.DataFrame(grace_rows)

    df_grace['date'] = pd.to_datetime(df_grace['date'])
    df_grace.set_index('date', inplace=True)
    df_grace.index = df_grace.index.to_period('M').to_timestamp()
    df_grace = df_grace[~df_grace.index.duplicated(keep='first')]
    df_grace['YearMonth'] = df_grace.index.to_period('M')

    # daily grid: each day inherits its month's GRACE value (GRACE is monthly);
    # months with no GRACE solution stay NaN
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_daily = pd.DataFrame({'date': daily_dates})
    df_daily['YearMonth'] = df_daily['date'].dt.to_period('M')

    final_df = df_daily.merge(df_grace[['YearMonth', 'GRACE_LWE_Anomaly_cm']], on='YearMonth', how='left')
    final_df.drop(columns=['YearMonth'], inplace=True)

    last_valid_date = df_grace.dropna(subset=['GRACE_LWE_Anomaly_cm']).index[-1]
    final_df = final_df[final_df['date'] <= last_valid_date + pd.Timedelta(days=31)]

    return final_df


def fill_gaps(df):
    """
    Fill missing values (NaN) in GRACE data using temporal linear interpolation.

    Parameters:
        df: a panda dataframe containing GRACE data

    Returns:
        df_filled: a panda dataframe with missing values filled by linear interpolation
    """

    df_filled = df.copy()
    df_filled['GRACE_LWE_Anomaly_cm'] = df_filled['GRACE_LWE_Anomaly_cm'].interpolate(method='linear')
    return df_filled


if __name__ == "__main__":
    project = os.environ.get("EARTHENGINE_PROJECT")

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    paca_names = list(PACA.values())

    output_dir = "data/grace"
    os.makedirs(output_dir, exist_ok=True)

    start_date = "2002-04-01"

    print(f"\n--- Processing GRACE (Start Date: {start_date}, daily) ---")

    for names, region_name in [(paca_names, "PACA")]:
        print(f"Fetching data for {region_name}...")
        df = get_grace_for_region(names, region_name, start_date=start_date)

        raw_path = os.path.join(output_dir, f"daily_raw_GRACE_{region_name}.csv")
        df.to_csv(raw_path, index=False)
        print(f"Saved raw data to {raw_path}")

        df_filled = fill_gaps(df)
        filled_path = os.path.join(output_dir, f"daily_filled_GRACE_{region_name}.csv")
        df_filled.to_csv(filled_path, index=False)
        print(f"Saved filled data to {filled_path}")

    print("\nDone! Aligned datasets generated in the 'data/grace' folder.")