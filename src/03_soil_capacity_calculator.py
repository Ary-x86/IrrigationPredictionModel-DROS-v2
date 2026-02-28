# src/03_soil_capacity_calculator.py

import pandas as pd
import numpy as np

def calculate_capacity_and_merge():
    print("Loading intermediate datasets...")
    
    # 1. Load the 10-minute sensor data and 1-hour weather data
    df_sensors = pd.read_csv('./data/merged_sensor_data.csv')
    df_weather = pd.read_csv('./data/open_meteo_forecast_data.csv')

    # Convert to datetime objects
    df_sensors['datetime'] = pd.to_datetime(df_sensors['datetime'])
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

    print("Merging 1-hour weather forecasts into 10-minute sensor data...")
    # 2. The Missing Link: Merge datasets
    # We round down the 10-minute sensor timestamps to the nearest hour to match the weather forecast
    df_sensors['join_hour'] = df_sensors['datetime'].dt.floor('h')
    df_weather['join_hour'] = df_weather['datetime']
    
    # Merge and drop the temporary join column
    df_final = pd.merge(df_sensors, df_weather, on='join_hour', how='inner')
    df_final = df_final.drop(columns=['join_hour', 'datetime_y'])
    df_final = df_final.rename(columns={'datetime_x': 'datetime'})

    print("Calculating Soil Capacity Point via Time Derivative...")
    # 3. Calculate the time derivative of soil moisture
    # The paper computes the time derivative of soil moisture to find where the decline rate slows down.
    df_final['Moisture_Derivative'] = df_final['Soil Moisture [RH%]'].diff()
    
    # 4. Estimate Field Capacity (mu) and Standard Deviation (sigma)
    # We isolate periods of slow drainage (e.g., negative derivative, but close to 0) 
    # to find the soil capacity point [cite: 170-171].
    slow_drainage_mask = (df_final['Moisture_Derivative'] < 0) & (df_final['Moisture_Derivative'] >= -0.5)
    capacity_data = df_final[slow_drainage_mask]['Soil Moisture [RH%]']
    
    mu = capacity_data.mean()
    sigma = capacity_data.std()
    print(f"Calculated Soil Capacity (mu): {mu:.2f}%")
    print(f"Calculated Standard Deviation (sigma): {sigma:.2f}%")

    print("Generating AI Target Classes (0=OFF, 1=ON, 2=No Adj, 3=Alert)...")
    # 5. Define the Confidence Intervals
    # The interval focuses on the lower limit of mu - sigma/2 and upper of mu + sigma/2[cite: 222].
    lower_limit_standard = mu - (sigma / 2)
    upper_limit_standard = mu + (sigma / 2)
    critical_limit = mu - sigma  # Used for the Alert state [cite: 253-254]

    # Initialize the target column
    df_final['Irrigation_Decision'] = 2 # Default to 'No Adjustment' (Class 2)

    # Apply the logic from the paper's flow diagram (Figure 3) [cite: 248, 259-264]
    for index, row in df_final.iterrows():
        moisture = row['Soil Moisture [RH%]']
        precip_forecast = row['Weather Forecast Rainfall [mm]']
        
        # Condition 1: Soil moisture < mu - sigma/2
        if moisture < lower_limit_standard:
            # Check for precipitation > 2 mm
            if precip_forecast > 2.0:
                # If rain is coming but moisture is critically low (below mu - sigma), send Alert (Class 3)
                if moisture < critical_limit:
                    df_final.at[index, 'Irrigation_Decision'] = 3
                else:
                    # Rain is coming, soil is low but not critical -> do nothing (Class 2)
                    df_final.at[index, 'Irrigation_Decision'] = 2
            else:
                # No rain coming, soil is dry -> Irrigation ON (Class 1)
                df_final.at[index, 'Irrigation_Decision'] = 1
                
        # Condition 2: Soil moisture > mu + sigma/2
        elif moisture > upper_limit_standard:
            # Soil is over capacity -> Irrigation OFF (Class 0)
            df_final.at[index, 'Irrigation_Decision'] = 0
            
        # If neither, it remains 2 (No Adjustment)

    # 6. Final Cleanup
    # Drop intermediate calculation columns and match the final dataset features [cite: 523-524]
    final_features = [
        'Soil Moisture [RH%]', 
        'Soil Temperature [C]', 
        'Environmental Temperature [ C]', 
        'Environmental Humidity [RH %]',
        'Weather Forecast Rainfall [mm]',
        'Crop Data Evapotranspiration [mm]',
        'Irrigation_Decision'
    ]
    df_final = df_final[final_features].dropna()

    print(f"Dataset finalized! Final shape: {df_final.shape}")
    
    # Save the absolute final dataset ready for Neural Network training!
    output_path = './data/processed_dataset.csv'
    df_final.to_csv(output_path, index=False)
    print(f"Successfully saved AI training data to {output_path}")

if __name__ == "__main__":
    calculate_capacity_and_merge()