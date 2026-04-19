"""Threshold policy on the forecaster's t+3h VWC prediction.

action = ON_HIGH if vwc_hat(t+3h) < MAD_lo
         ON_LOW  if vwc_hat(t+3h) < MAD_hi
         OFF     otherwise

MAD is stage-aware per FAO-56 (MAD_FRACTION_BY_STAGE).
Backing: MPC-style rule policy over a forecaster (Ikegawa 2026, WIAI 2021).
This is the simple static version; the MPC refinement lands in Track B Phase 3b.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.gbdt.config import (
    ACTION_VOLUME_MM,
    FIELD_CAPACITY_PCT,
    MAD_FRACTION_BY_STAGE,
    WILTING_POINT_PCT,
)


def mad_threshold(stage: str) -> float:
    frac = MAD_FRACTION_BY_STAGE.get(stage, 0.40)
    taw = FIELD_CAPACITY_PCT - WILTING_POINT_PCT
    # readily available water floor = FC - MAD * TAW
    return FIELD_CAPACITY_PCT - frac * taw


def decide(vwc_hat_3h: np.ndarray, stages: pd.Series) -> pd.DataFrame:
    mad_hi = stages.map(mad_threshold).astype(float).values
    mad_lo = mad_hi - 2.0  # below this -> high volume

    action = np.full(len(vwc_hat_3h), "OFF", dtype=object)
    action[vwc_hat_3h < mad_hi] = "ON_LOW"
    action[vwc_hat_3h < mad_lo] = "ON_HIGH"

    volume = np.array([ACTION_VOLUME_MM[a] for a in action], dtype=float)
    return pd.DataFrame({
        "vwc_hat_3h": vwc_hat_3h,
        "mad_hi": mad_hi,
        "mad_lo": mad_lo,
        "action": action,
        "volume_mm": volume,
    })
