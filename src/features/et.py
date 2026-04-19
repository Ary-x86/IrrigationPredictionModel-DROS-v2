"""Evapotranspiration features: VPD, Hargreaves-Samani ET0, ETc.

- VPD (Tetens / Magnus form).
- Hargreaves-Samani daily ET0 as a backup when Penman-Monteith inputs are
  unavailable (e.g., only Tmin/Tmax + geographic radiation).
- ETc = kc * ET0 for the crop-water-demand feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vpd_kpa(t_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Vapor Pressure Deficit in kPa. 0 when air saturated."""
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    ea = es * rh_pct / 100.0
    return (es - ea).clip(lower=0.0)


def hargreaves_et0_mm_day(
    t_mean: pd.Series,
    t_max: pd.Series,
    t_min: pd.Series,
    ra_mj_m2_day: np.ndarray,
) -> pd.Series:
    """FAO-56 Hargreaves-Samani. ra_mj_m2_day is extraterrestrial radiation."""
    delta_t = (t_max - t_min).clip(lower=0.0)
    return 0.0023 * (t_mean + 17.8) * np.sqrt(delta_t) * ra_mj_m2_day * 0.408


def etc_mm_per_hour(kc: pd.Series, et0_mm_per_hour: pd.Series) -> pd.Series:
    """Crop evapotranspiration per hour. Inputs must share time resolution."""
    return (kc * et0_mm_per_hour).clip(lower=0.0)
