"""Soil-moisture-derived features: lags, rolling stats, derivatives, SWDI.

SWDI = Soil Water Deficit Index = (VWC - WP) / (FC - WP).
1.0 = at field capacity, 0.0 = at wilting point, negative means below WP.
"""
from __future__ import annotations

import pandas as pd

from src.config import FIELD_CAPACITY_PCT, WILTING_POINT_PCT

VWC_LAGS_STEPS = [6, 18, 36, 72, 144]
ROLL_MEAN_STEPS = 18   # 3h at 10-min cadence
ROLL_STD_STEPS = 36    # 6h at 10-min cadence
DERIV_STEPS = 6        # 1h finite difference


def add_vwc_features(df: pd.DataFrame, group_col: str, vwc_col: str) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(group_col, sort=False)[vwc_col]
    for h in VWC_LAGS_STEPS:
        out[f"{vwc_col}_lag_{h}s"] = g.shift(h)
    out[f"{vwc_col}_roll_mean_3h"] = g.transform(lambda s: s.rolling(ROLL_MEAN_STEPS, min_periods=1).mean())
    out[f"{vwc_col}_roll_std_6h"] = g.transform(lambda s: s.rolling(ROLL_STD_STEPS, min_periods=2).std()).fillna(0.0)
    out[f"{vwc_col}_deriv_1h"] = g.transform(lambda s: s.diff(DERIV_STEPS)).fillna(0.0)
    return out


def swdi(vwc_series: pd.Series) -> pd.Series:
    return (vwc_series - WILTING_POINT_PCT) / (FIELD_CAPACITY_PCT - WILTING_POINT_PCT)
