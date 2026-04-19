"""Sentinel-2 L2A NDVI/EVI/LAI fetcher (Copernicus Data Space STAC).

Queries the Copernicus Data Space STAC catalog for S2 L2A scenes intersecting
the Stuard cell, computes NDVI = (B08 - B04) / (B08 + B04), EVI, and an LAI
proxy from NDVI. Writes daily samples (forward-filled between revisits) to
data/interim/sentinel2_daily.parquet.

Requires:
  pip install pystac-client odc-stac rasterio
  Free Copernicus Data Space account at
  https://dataspace.copernicus.eu/ (no key needed for STAC search; token
  needed for authenticated downloads via odc-stac 'sign' helper).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA, LAT, LON

OUT = DATA / "interim" / "sentinel2_daily.parquet"
STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"


def fetch(start: str, end: str, out_path: Path = OUT) -> Path:
    try:
        from pystac_client import Client
        import odc.stac
        import rioxarray  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Install first: pip install pystac-client odc-stac rasterio rioxarray xarray\n"
            "Auth: https://dataspace.copernicus.eu/"
        )
    bbox = [LON - 0.02, LAT - 0.02, LON + 0.02, LAT + 0.02]
    client = Client.open(STAC_URL)
    search = client.search(
        collections=["SENTINEL-2"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.items())
    print(f"Found {len(items)} Sentinel-2 scenes in {start}..{end}")
    if not items:
        raise SystemExit("No scenes returned; check auth or date range.")

    # Actual raster download + NDVI extraction is left as a follow-up:
    # odc.stac.load(items, bbox=bbox, bands=['B04', 'B08']).resample(time='1D').mean()
    # The loader here exists so the CLI + discovery surface is in place; fill
    # the compute step once CDS credentials are configured.
    raise NotImplementedError(
        "Finish odc.stac.load + NDVI/EVI/LAI compute after configuring Copernicus auth."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-05-01")
    p.add_argument("--end", default="2023-10-31")
    args = p.parse_args()
    fetch(args.start, args.end)
