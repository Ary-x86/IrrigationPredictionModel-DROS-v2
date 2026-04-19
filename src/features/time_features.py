"""Cyclical time encodings: sin/cos of day-of-year and hour-of-day.

Use cyclical encoding so the model sees Dec 31 and Jan 1 as neighbors, and
hour 23 as close to hour 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    t = pd.to_datetime(df[time_col])
    doy = t.dt.dayofyear.to_numpy()
    hour = t.dt.hour.to_numpy() + t.dt.minute.to_numpy() / 60.0
    out = df.copy()
    out["doy"] = doy
    out["hour"] = hour
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.0)
    out["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
    return out
