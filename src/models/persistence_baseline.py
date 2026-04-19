"""Persistence baseline: y_hat(t+h) = vwc(t) for every horizon.

Strong on stable weeks — any forecaster that cannot beat this is useless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HORIZON_LABELS


def predict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    base = df["vwc_20cm"].to_numpy()
    return {label: base.copy() for label in HORIZON_LABELS}
