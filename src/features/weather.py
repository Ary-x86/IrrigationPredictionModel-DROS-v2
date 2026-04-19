"""Weather-side features: rolling rain sums, hours-since-last-irrigation."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAIN_SUM_WINDOWS = {"rain_mm_1h_sum": 6, "rain_mm_6h_sum": 36, "rain_mm_24h_sum": 144}


def add_rain_rolls(df: pd.DataFrame, group_col: str, rain_col: str) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(group_col, sort=False)[rain_col]
    for name, steps in RAIN_SUM_WINDOWS.items():
        out[name] = g.transform(lambda s, n=steps: s.rolling(n, min_periods=1).sum())
    return out


def hours_since_last_irrigation(df: pd.DataFrame, group_col: str, time_col: str, irr_col: str) -> pd.Series:
    """Hours since the last positive irrigation volume per line."""
    out = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby(group_col, sort=False):
        ts = g[time_col].to_numpy()
        irr = g[irr_col].fillna(0.0).to_numpy() > 0.0
        last_ts = None
        hours = np.full(len(g), np.nan, dtype=float)
        for i in range(len(g)):
            if irr[i]:
                last_ts = ts[i]
            if last_ts is not None:
                hours[i] = (ts[i] - last_ts) / np.timedelta64(1, "h")
        out.loc[g.index] = hours
    if out.notna().any():
        return out.fillna(out.median())
    return out.fillna(0.0)
