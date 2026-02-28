# src/02_data_preprocessing.py

import pandas as pd
import numpy as np

def load_and_clean_data():
    print("Loading raw CSV files...")

    # 1. Load the three datasets
    df_env = pd.read_csv('./data/stuard_environmental_data.csv')
    df_soil = pd.read_csv('./data/stuard_soil_data.csv')
    df_water = pd.read_csv('./data/stuard_water_meter_data.csv')

    # FIX 1: Clean accidental header-rows inside the data
    df_env = df_env[df_env['ts_generation'] != 'ts_generation']
    # 2. Filter for Line #1 ONLY
    df_soil = df_soil[df_soil['ts_generation'] != 'ts_generation']
    df_water = df_water[df_water['ts_generation'] != 'ts_generation']

    # FIX 2: Force 'line' to be a string and then filter
    df_soil['line'] = df_soil['line'].astype(str)
    df_water['line'] = df_water['line'].astype(str)
    
    df_soil = df_soil[df_soil['line'] == '1'].copy()
    df_water = df_water[df_water['line'] == '1'].copy()

    # 3. Format Timestamps
    # FIX 3: Convert timestamps and metrics to numeric (ignoring errors)
    for df in [df_env, df_soil, df_water]:
        df['ts_generation'] = pd.to_numeric(df['ts_generation'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['ts_generation'], unit='ms')
        df['datetime'] = df['datetime'].dt.round('10min')

    # 4. Select and rename Environmental columns
    print("Processing Environmental Data...")
    df_env['temperature'] = pd.to_numeric(df_env['temperature'], errors='coerce')
    df_env['humidity'] = pd.to_numeric(df_env['humidity'], errors='coerce')
    df_env = df_env[['datetime', 'temperature', 'humidity']]
    df_env.columns = ['datetime', 'Environmental Temperature [ C]', 'Environmental Humidity [RH %]']
    df_env = df_env.drop_duplicates(subset='datetime')

    print("Processing Soil Data...")
    # 5. Select and rename Soil columns
    df_soil['humidity'] = pd.to_numeric(df_soil['humidity'], errors='coerce')
    df_soil['temperature'] = pd.to_numeric(df_soil['temperature'], errors='coerce')
    df_soil['electrical_conductivity'] = pd.to_numeric(df_soil['electrical_conductivity'], errors='coerce')
    
    df_soil = df_soil[['datetime', 'humidity', 'temperature', 'electrical_conductivity']]
    df_soil.columns = ['datetime', 'Soil Moisture [RH%]', 'Soil Temperature [C]', 'Soil Electrical Conductivity']
    df_soil = df_soil.drop_duplicates(subset='datetime')

    print("Processing Water Meter Data...")
    # 6. Calculate Irrigation ON/OFF using the 'current_volume' column
    df_water['current_volume'] = pd.to_numeric(df_water['current_volume'], errors='coerce')
    df_water = df_water[['datetime', 'current_volume']].sort_values('datetime')
    df_water = df_water.drop_duplicates(subset='datetime')
    df_water['volume_diff'] = df_water['current_volume'].diff()
    df_water['Irrigation (ON/OFF)'] = np.where(df_water['volume_diff'] > 0, 1, 0)
    df_water = df_water[['datetime', 'Irrigation (ON/OFF)']]

    print("Merging datasets into a single dataframe...")
    # Use outer merge + ffill to handle imperfect sensor timing
    # 7. Merge everything together using an OUTER merge
    df_merged = pd.merge(df_env, df_soil, on='datetime', how='outer')
    df_merged = pd.merge(df_merged, df_water, on='datetime', how='outer')
    df_merged = df_merged.sort_values('datetime').ffill().dropna()

    # Sort by time, carry forward the last known sensor readings to fill gaps, then drop remaining NaNs
     # 8. Extract 'Daily Hour'
    df_merged['Daily Hour'] = df_merged['datetime'].dt.hour

    # 9. Final Reorder
    final_columns = ['datetime', 'Irrigation (ON/OFF)', 'Soil Moisture [RH%]', 'Soil Temperature [C]', 'Soil Electrical Conductivity', 'Daily Hour', 'Environmental Temperature [ C]', 'Environmental Humidity [RH %]']
    df_merged = df_merged[final_columns]

    print(f"Data processing complete! Final dataset shape: {df_merged.shape}")
    df_merged.to_csv('./data/merged_sensor_data.csv', index=False)

if __name__ == "__main__":
    load_and_clean_data()