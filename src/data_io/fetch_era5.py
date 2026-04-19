"""ERA5 reanalysis fetcher (Copernicus CDS API).

Pulls hourly t2m, d2m (→ RH), tp (precip), ssrd (solar), u10/v10 (wind) for
a bounding box around the Stuard cell for 2020-01-01..2023-10-31. Output is a
single netCDF per year under data/raw/era5/, plus a concatenated parquet at
data/interim/era5_hourly.parquet.

Requires:
  1. Copernicus CDS account at https://cds.climate.copernicus.eu/
  2. ~/.cdsapirc with 'url' and 'key' (see CDS portal -> user profile)
  3. `pip install cdsapi xarray netCDF4`

Run:
  python -m src.data_io.fetch_era5 --year 2020
  python -m src.data_io.fetch_era5 --year 2021
  ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA, LAT, LON

RAW = DATA / "raw" / "era5"

BBOX = [LAT + 0.25, LON - 0.25, LAT - 0.25, LON + 0.25]  # N, W, S, E
VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def fetch_year(year: int, out_dir: Path = RAW) -> Path:
    try:
        import cdsapi
    except ImportError:
        raise SystemExit(
            "cdsapi not installed. Run: pip install cdsapi xarray netCDF4\n"
            "Also create ~/.cdsapirc with your key from "
            "https://cds.climate.copernicus.eu/user/<id>"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"era5_{year}.nc"
    if out_path.exists():
        print(f"{out_path} exists; skipping")
        return out_path
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": VARS,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": BBOX,
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    args = p.parse_args()
    fetch_year(args.year)
