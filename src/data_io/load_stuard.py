"""Load the Stuard sensor CSV + Open-Meteo hourly forecast and join them.

Emits a per-line 10-min grid with a single datetime index, sensor VWC, soil
and air T, RH, irrigation volume_diff, and joined hourly forecast columns.
"""
from __future__ import annotations

import pandas as pd

from src.config import DATA

SENSOR_CSV = DATA / "merged_sensor_data.csv"
WEATHER_CSV = DATA / "open_meteo_forecast_data.csv"


def load_and_join() -> pd.DataFrame:
    sensors = pd.read_csv(SENSOR_CSV)
    weather = pd.read_csv(WEATHER_CSV)
    sensors["datetime"] = pd.to_datetime(sensors["datetime"])
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    sensors["line"] = sensors["line"].astype(int)

    sensors["join_hour"] = sensors["datetime"].dt.floor("h")
    weather = weather.rename(columns={"datetime": "join_hour"})
    joined = sensors.merge(weather, on="join_hour", how="left").drop(columns=["join_hour"])
    joined = joined.sort_values(["line", "datetime"]).reset_index(drop=True)
    return joined
