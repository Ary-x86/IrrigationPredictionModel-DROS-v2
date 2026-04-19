"""Rule policy on forecasted VWC. Phase-3a.

Reads the t+3h VWC prediction and the current growth stage, applies a
stage-aware Management Allowed Depletion (MAD) threshold, and emits
{OFF, ON_LOW, ON_HIGH}. Equivalent to Track A's `experiments/gbdt/policy_rule.py`
but hosted in `src/policy/` for the Track B pipeline.
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


def mad_threshold(stage: str) -> float:
    frac = MAD_FRACTION_BY_STAGE.get(stage, 0.40)
    taw = FIELD_CAPACITY_PCT - WILTING_POINT_PCT
    return FIELD_CAPACITY_PCT - frac * taw


def decide(vwc_hat_3h: np.ndarray, stages: pd.Series) -> pd.DataFrame:
    mad_hi = stages.map(mad_threshold).astype(float).values
    mad_lo = mad_hi - 2.0
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
