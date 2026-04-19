"""Forecasting baselines for Track A.

- Persistence:    y_hat(t+h) = vwc(t)
- Climatology:    per-line mean VWC on train
- Water-balance:  dM/dt = rain - etc  (simple bucket, no irrigation knowledge)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.gbdt.config import HORIZON_LABELS


def persistence_forecast(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for label in HORIZON_LABELS:
        out[f"pred_h{label}"] = df["vwc_20cm"].values
    return out


def climatology_forecast(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    means = train_df.groupby("line")["vwc_20cm"].mean()
    per_row = eval_df["line"].map(means).astype(float).values
    out = pd.DataFrame(index=eval_df.index)
    for label in HORIZON_LABELS:
        out[f"pred_h{label}"] = per_row
    return out


def water_balance_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Naive bucket over the forecast horizon: integrate rain - etc from t."""
    out = pd.DataFrame(index=df.index)
    rain = df["rain_mm_h"].fillna(0.0).values
    etc = df["etc_mm_h"].fillna(0.0).values
    base = df["vwc_20cm"].values
    horizon_hours = {"1h": 1, "3h": 3, "6h": 6, "12h": 12, "24h": 24}
    for label, h in horizon_hours.items():
        delta = (rain - etc) * h
        out[f"pred_h{label}"] = np.clip(base + delta, 0.0, 100.0)
    return out
