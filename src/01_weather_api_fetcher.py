# src/01_weather_api_fetcher.py

from pathlib import Path
import requests
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fetch_weather_data():
    print("Preparing to fetch weather data from Open Meteo API...")

    # 1. Use the exact coordinates and reference period described in the paper.
    # We are reconstructing the original experiment as closely as possible.
    latitude = 44.1125
    longitude = 10.411
    start_date = "2023-07-28"
    end_date = "2023-09-03"
    timezone = "Europe/Rome"

    # 2. Historical archive endpoint.
    # The paper uses hourly precipitation, forecast humidity, and reference evapotranspiration.
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "relative_humidity_2m,precipitation,et0_fao_evapotranspiration",
        "timezone": timezone,
    }

    print(f"Fetching data from {start_date} to {end_date} for coordinates {latitude}, {longitude}...")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise KeyError("Open-Meteo response does not contain an 'hourly' section.")

    hourly = payload["hourly"]

    # 3. Parse into a clean DataFrame.
    df_weather = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"]).tz_localize(None),
            "Weather Forecast Rainfall [mm]": pd.to_numeric(hourly["precipitation"], errors="coerce"),
            "Weather Forecast Environmental humidity [RH %]": pd.to_numeric(
                hourly["relative_humidity_2m"], errors="coerce"
            ),
            "Crop Data Evapotranspiration [mm]": pd.to_numeric(
                hourly["et0_fao_evapotranspiration"], errors="coerce"
            ),
        }
    )

    # 4. Sanity cleanup.
    df_weather = (
        df_weather.sort_values("datetime")
        .drop_duplicates(subset="datetime")
        .reset_index(drop=True)
    )

    print("Data successfully structured!")
    print(df_weather.head())
    print(f"Total weather records fetched: {len(df_weather)}")

    # 5. Save for downstream merge.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "open_meteo_forecast_data.csv"
    df_weather.to_csv(output_path, index=False)
    print(f"Saved weather data to {output_path}")


if __name__ == "__main__":
    fetch_weather_data()