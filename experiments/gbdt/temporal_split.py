"""Temporal splits with 24h embargo for Track A.

Strict date-range splits with an embargo gap on both sides of val/test to
kill adjacency leakage across the boundary.
"""
from __future__ import annotations

import pandas as pd

from experiments.gbdt.config import (
    EMBARGO_HOURS,
    EXP_DATA,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)


def split_by_time(df: pd.DataFrame, time_col: str = "datetime") -> dict[str, pd.DataFrame]:
    t = pd.to_datetime(df[time_col])
    emb = pd.Timedelta(hours=EMBARGO_HOURS)

    train_lo, train_hi = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    val_lo, val_hi = pd.Timestamp(VAL_START), pd.Timestamp(VAL_END)
    test_lo, test_hi = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)

    train_mask = (t >= train_lo) & (t <= train_hi - emb)
    val_mask = (t >= val_lo + emb) & (t <= val_hi - emb)
    test_mask = (t >= test_lo + emb) & (t <= test_hi)

    return {
        "train": df.loc[train_mask].copy().reset_index(drop=True),
        "val": df.loc[val_mask].copy().reset_index(drop=True),
        "test": df.loc[test_mask].copy().reset_index(drop=True),
    }


def load_features() -> pd.DataFrame:
    path = EXP_DATA / "features.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


if __name__ == "__main__":
    df = load_features()
    parts = split_by_time(df)
    for name, part in parts.items():
        lo = part["datetime"].min()
        hi = part["datetime"].max()
        print(f"{name}: {len(part):,} rows  [{lo} .. {hi}]")
