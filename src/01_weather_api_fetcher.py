import requests
import pandas as pd

def fetch_weather_data():
    print("Preparing to fetch weather data from Open Meteo API...")

    # 1. Define the exact parameters specified in the research paper
    # The authors noted their API request used these specific coordinates.
    # (Note: These are slightly offset from the exact farm coordinates, but we use them to perfectly replicate the paper's API call).
    latitude = 44.1125
    longitude = 10.411
    
    # The reference period for the dataset is July 28th to September 3rd, 2023.
    start_date = "2023-07-28"
    end_date = "2023-09-03"
    
    # European time zone (GMT+1), which for Italy is Europe/Rome.
    timezone = "Europe/Rome"

    # 2. Set up the Open Meteo API request
    # We use the historical archive API since we are reconstructing a past experiment.
    # The paper requires hourly precipitation, relative humidity, and evapotranspiration reference rate.
    # Open Meteo uses 'et0_fao_evapotranspiration' for the Penman-Monteith reference evapotranspiration[cite: 174].
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "relative_humidity_2m,precipitation,et0_fao_evapotranspiration",
        "timezone": timezone
    }

    print(f"Fetching data from {start_date} to {end_date} for coordinates {latitude}, {longitude}...")
    
    # 3. Make the API call
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        print(response.text)
        return

    data = response.json()

    # 4. Parse the JSON response into a Pandas DataFrame
    print("Parsing JSON response into a dataset...")
    hourly_data = data["hourly"]
    
    df_weather = pd.DataFrame({
        "datetime": pd.to_datetime(hourly_data["time"]),
        "Weather Forecast Rainfall [mm]": hourly_data["precipitation"],
        "Weather Forecast Environmental humidity [RH %]": hourly_data["relative_humidity_2m"],
        "Crop Data Evapotranspiration [mm]": hourly_data["et0_fao_evapotranspiration"]
    })

    # 5. Review the fetched data
    # The dataset should contain hourly records for the entire date range.
    print("Data successfully structured!")
    print(df_weather.head())
    print(f"Total weather records fetched: {len(df_weather)}")

    # 6. Save the data to the 'data/' folder
    # We will use this CSV in our next step to merge with the sensor data.
    output_path = "./data/open_meteo_forecast_data.csv"
    df_weather.to_csv(output_path, index=False)
    print(f"Saved weather data to {output_path}")

if __name__ == "__main__":
    fetch_weather_data()