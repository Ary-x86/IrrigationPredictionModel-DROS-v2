"""Temporal splits with 24h embargo. Shared by Track B training + eval."""
from __future__ import annotations

import pandas as pd

from src.config import (
    EMBARGO_HOURS,
    TEST_END, TEST_START,
    TRAIN_END, TRAIN_START,
    VAL_END, VAL_START,
)


def split(df: pd.DataFrame, time_col: str = "datetime") -> dict[str, pd.DataFrame]:
    t = pd.to_datetime(df[time_col])
    emb = pd.Timedelta(hours=EMBARGO_HOURS)
    tr_lo, tr_hi = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    va_lo, va_hi = pd.Timestamp(VAL_START), pd.Timestamp(VAL_END)
    te_lo, te_hi = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    masks = {
        "train": (t >= tr_lo) & (t <= tr_hi - emb),
        "val": (t >= va_lo + emb) & (t <= va_hi - emb),
        "test": (t >= te_lo + emb) & (t <= te_hi),
    }
    return {k: df.loc[m].copy().reset_index(drop=True) for k, m in masks.items()}
