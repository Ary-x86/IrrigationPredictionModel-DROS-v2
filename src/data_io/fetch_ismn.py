"""International Soil Moisture Network fetcher.

ISMN publishes in-situ VWC from ~2700 stations for cross-site pretraining.
Italian/Spanish tomato-region stations are the closest priors for Stuard.

Flow:
  1. Register at https://ismn.earth/en/ (free, manual approval ~1 day)
  2. Request a download bundle from the portal (web UI, filter by network +
     time range). The portal emails a zip link.
  3. Unpack the zip into data/raw/ismn/; run this script to convert to a
     tidy parquet at data/interim/ismn_hourly.parquet.

Requires: pip install ismn
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA

RAW = DATA / "raw" / "ismn"
OUT = DATA / "interim" / "ismn_hourly.parquet"


def convert(raw_dir: Path = RAW, out_path: Path = OUT) -> Path:
    try:
        from ismn.interface import ISMN_Interface
    except ImportError:
        raise SystemExit("Install first: pip install ismn")
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise SystemExit(
            f"No ISMN bundle at {raw_dir}. Download from https://ismn.earth/ first."
        )
    iface = ISMN_Interface(str(raw_dir))
    print(f"Networks: {iface.list_networks()}")
    raise NotImplementedError(
        "Finish network/station iteration + tidy DataFrame build once the "
        "ISMN bundle for the target networks lives at data/raw/ismn/."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=str(RAW))
    args = p.parse_args()
    convert(Path(args.raw_dir))
