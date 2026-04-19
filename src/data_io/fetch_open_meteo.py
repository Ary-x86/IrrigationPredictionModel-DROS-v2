"""Fetch Open-Meteo hourly historical forecast for the Stuard cell.

Free API, no key needed. The existing `data/open_meteo_forecast_data.csv` was
produced by the legacy `src/01_weather_api_fetcher.py`; this module is the
refactored entrypoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA, LAT, LON

OUT = DATA / "open_meteo_forecast_data.csv"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "et0_fao_evapotranspiration",
]


def fetch(start: str, end: str, out_path: Path = OUT) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Rome",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data).rename(columns={
        "time": "datetime",
        "precipitation": "Weather Forecast Rainfall [mm]",
        "et0_fao_evapotranspiration": "Crop Data Evapotranspiration [mm]",
        "relative_humidity_2m": "Weather Forecast Environmental humidity [RH %]",
        "temperature_2m": "Weather Forecast Temperature [C]",
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-05-01")
    p.add_argument("--end", default="2023-10-31")
    args = p.parse_args()
    fetch(args.start, args.end)
