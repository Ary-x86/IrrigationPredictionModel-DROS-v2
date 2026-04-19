"""Derive future-VWC targets per horizon. No threshold-derived class labels.

y_vwc_h{1h,3h,6h,12h,24h}(t) = vwc(t + horizon)  per irrigation line.
"""
from __future__ import annotations

import pandas as pd

from src.config import HORIZONS_10MIN, HORIZON_LABELS


def add_future_vwc(df: pd.DataFrame, group_col: str, vwc_col: str) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(group_col, sort=False)[vwc_col]
    for steps, label in zip(HORIZONS_10MIN, HORIZON_LABELS):
        out[f"y_vwc_h{label}"] = g.shift(-steps)
    return out


def target_columns() -> list[str]:
    return [f"y_vwc_h{label}" for label in HORIZON_LABELS]
