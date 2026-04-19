"""Growing degree days, growth stage, dynamic Kc.

GDD is cumulative heat summation from transplant date using base T=10°C.
Stage edges and per-stage Kc come from config (FAO-56 tomato values).
Kc is piecewise-linear across stage centers so the curve is smooth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GDD_STAGE_EDGES, KC_BY_STAGE, T_BASE_C, TRANSPLANT_DATE


def gdd_cumulative(datetime_series: pd.Series, t_air_c: pd.Series) -> pd.Series:
    transplant = pd.Timestamp(TRANSPLANT_DATE).date()
    dates = pd.to_datetime(datetime_series).dt.date
    per_day_mean = pd.Series(t_air_c.values, index=dates).groupby(level=0).mean()
    per_day_gdd = (per_day_mean - T_BASE_C).clip(lower=0.0).sort_index()
    per_day_gdd = per_day_gdd[per_day_gdd.index >= transplant]
    cum = per_day_gdd.cumsum().to_dict()
    return dates.map(cum).astype(float)


def stage_from_gdd(gdd: float) -> str:
    for lo, hi, name in GDD_STAGE_EDGES:
        if lo <= gdd < hi:
            return name
    return GDD_STAGE_EDGES[-1][2]


def kc_from_gdd(gdd: float) -> float:
    centers = [(lo + hi) / 2.0 for lo, hi, _ in GDD_STAGE_EDGES]
    kcs = [KC_BY_STAGE[name] for _, _, name in GDD_STAGE_EDGES]
    if gdd <= centers[0]:
        return kcs[0]
    if gdd >= centers[-1]:
        return kcs[-1]
    return float(np.interp(gdd, centers, kcs))
