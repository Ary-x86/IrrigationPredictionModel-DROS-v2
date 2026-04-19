"""NASA SMAP L3/L4 rootzone soil moisture fetcher.

SMAP L4 provides 9km, 3-hourly rootzone VWC worldwide — a satellite prior
for the Stuard root zone useful during transfer-learning pretraining.

Flow:
  1. Register at https://urs.earthdata.nasa.gov/ (free; accept NASA EULAs).
  2. Create ~/.netrc with 'machine urs.earthdata.nasa.gov login <user>
     password <pass>'.
  3. Run this script with --start/--end; it uses earthaccess to query the
     CMR catalog and download the granules to data/raw/smap/.

Requires: pip install earthaccess xarray h5py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA, LAT, LON

RAW = DATA / "raw" / "smap"


def fetch(start: str, end: str, out_dir: Path = RAW) -> Path:
    try:
        import earthaccess
    except ImportError:
        raise SystemExit("Install first: pip install earthaccess xarray h5py")
    out_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login(strategy="netrc")
    results = earthaccess.search_data(
        short_name="SPL4SMAU",  # L4 analysis-update (rootzone SM)
        temporal=(start, end),
        bounding_box=(LON - 0.25, LAT - 0.25, LON + 0.25, LAT + 0.25),
    )
    print(f"Found {len(results)} SMAP L4 granules")
    if not results:
        raise SystemExit("No granules returned; check Earthdata login.")
    earthaccess.download(results, str(out_dir))
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2023-10-31")
    args = p.parse_args()
    fetch(args.start, args.end)
