"""Water-balance bucket baseline: vwc + h * (rain - etc).

Integrates rain - crop ET over each horizon. Known to blow up at long
horizons without irrigation knowledge; included so the LSTM has to beat a
physics-based reference to claim relevance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HORIZON_LABELS

_HORIZON_HOURS = {"1h": 1, "3h": 3, "6h": 6, "12h": 12, "24h": 24}


def predict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    base = df["vwc_20cm"].to_numpy()
    rain = df["rain_mm_h"].fillna(0.0).to_numpy()
    etc = df["etc_mm_h"].fillna(0.0).to_numpy()
    out: dict[str, np.ndarray] = {}
    for label in HORIZON_LABELS:
        dt = _HORIZON_HOURS[label]
        out[label] = np.clip(base + (rain - etc) * dt, 0.0, 100.0)
    return out
