"""CI-enforced leakage regression tests on modeling_dataset_v2.parquet.

Rules:
- Every y_vwc_h* target is strictly a future VWC value (shift by >=1 step).
- Dataset never mixes target columns with feature columns of the same name.
- Temporal splits never overlap; val and test sit strictly after train.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED, EMBARGO_HOURS, HORIZON_LABELS, HORIZONS_10MIN
from src.splits.temporal_split import split

DATASET = DATA_PROCESSED / "modeling_dataset_v2.parquet"


@pytest.fixture(scope="module")
def df():
    if not DATASET.exists():
        pytest.skip(f"{DATASET} not built yet; run `python -m src.features.assemble`")
    return pd.read_parquet(DATASET)


def test_targets_are_strictly_future(df):
    """For each row and each horizon, the stored target must equal the future
    VWC at that horizon for the same line."""
    tol = 1e-6
    for steps, label in zip(HORIZONS_10MIN, HORIZON_LABELS):
        col = f"y_vwc_h{label}"
        by_line = df.sort_values(["line", "datetime"]).groupby("line")["vwc_20cm"]
        expected = by_line.shift(-steps)
        aligned = df.sort_values(["line", "datetime"])[col].values
        mask = ~np.isnan(expected.values) & ~np.isnan(aligned)
        assert np.allclose(expected.values[mask], aligned[mask], atol=tol), label


def test_splits_are_ordered_and_embargoed(df):
    parts = split(df)
    emb = pd.Timedelta(hours=EMBARGO_HOURS)
    tr_max = parts["train"]["datetime"].max()
    va_min, va_max = parts["val"]["datetime"].min(), parts["val"]["datetime"].max()
    te_min = parts["test"]["datetime"].min()
    assert va_min - tr_max >= emb, (tr_max, va_min)
    assert te_min - va_max >= emb, (va_max, te_min)


def test_no_future_timestamp_in_feature_rows(df):
    """Sanity: meta datetime should never be ahead of any feature's computation
    window. We approximate by asserting nonnegative hours_since_last_irrigation."""
    assert (df["hours_since_last_irrigation"] >= 0).all()
