"""FAO-56 ETc-threshold scheduler baseline.

Canonical rule: irrigate when cumulative ETc since last watering meets the
refill depth for the current growth stage.

Inputs per decision step (10-min cadence):
  et0_mm_per_hour, kc_dynamic, vwc_20cm, growth_stage, time_since_last_mm

Decision: trigger ON_HIGH when the soil's allowable depletion (MAD * TAW) is
consumed; ON_LOW at half of that; OFF otherwise. Backing: Allen FAO-56.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ACTION_VOLUME_MM,
    FIELD_CAPACITY_PCT,
    MAD_FRACTION_BY_STAGE,
    WILTING_POINT_PCT,
)


def _mad_refill_mm(stage: str) -> float:
    frac = MAD_FRACTION_BY_STAGE.get(stage, 0.40)
    taw_pct = FIELD_CAPACITY_PCT - WILTING_POINT_PCT
    return frac * taw_pct


def decide(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise FAO-56 threshold decision.

    A real scheduler accumulates ETc over time; here we emit a decision per
    10-min step using the cumulative ETc since the most recent irrigation on
    the same line (computed from `hours_since_last_irrigation` × `etc_mm_h`).
    """
    stage = df["growth_stage"].astype(str)
    refill = stage.map(_mad_refill_mm).astype(float).values
    etc_mm_h = df["etc_mm_h"].fillna(0.0).to_numpy()
    hrs = df["hours_since_last_irrigation"].fillna(0.0).to_numpy()
    depleted_mm = np.clip(etc_mm_h * hrs, 0.0, None)

    action = np.full(len(df), "OFF", dtype=object)
    action[depleted_mm >= 0.5 * refill] = "ON_LOW"
    action[depleted_mm >= refill] = "ON_HIGH"
    volume = np.array([ACTION_VOLUME_MM[a] for a in action], dtype=float)
    return pd.DataFrame({
        "depleted_mm": depleted_mm,
        "refill_threshold_mm": refill,
        "action": action,
        "volume_mm": volume,
    })
