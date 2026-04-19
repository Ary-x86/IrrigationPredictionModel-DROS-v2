"""Sequence dataset for the LSTM forecaster.

Slides a 144-step (24h) lookback window over each line's timeline; labels
are the five future-VWC horizons already baked into modeling_dataset_v2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import HORIZON_LABELS, LOOKBACK_STEPS

SEQ_FEATURE_COLS = [
    "vwc_20cm",
    "vwc_20cm_deriv_1h",
    "soil_temp_c", "air_temp_c", "rh_pct", "vpd_kpa",
    "rain_mm_h", "rain_mm_1h_sum", "rain_mm_6h_sum", "rain_mm_24h_sum",
    "et0_open_meteo_mm_h", "etc_mm_h",
    "kc_dynamic",
    "sin_doy", "cos_doy", "sin_hour", "cos_hour",
    "hours_since_last_irrigation",
]


@dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray
    columns: list[str]

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def compute_stats(train_df: pd.DataFrame) -> FeatureStats:
    X = train_df[SEQ_FEATURE_COLS].to_numpy(dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return FeatureStats(mean=mean, std=std, columns=list(SEQ_FEATURE_COLS))


class VWCSequenceDataset(Dataset):
    """One sample per valid time index, per line.

    Each sample:
      seq:     (LOOKBACK_STEPS, F) standardized features
      line_id: int
      y:       (5,) future VWC at h={1,3,6,12,24}h
    """

    def __init__(self, df: pd.DataFrame, stats: FeatureStats) -> None:
        self.stats = stats
        self.target_cols = [f"y_vwc_h{h}" for h in HORIZON_LABELS]
        samples: list[dict] = []
        for line_id, group in df.groupby("line_id", sort=False):
            group = group.sort_values("datetime").reset_index(drop=True)
            X = group[SEQ_FEATURE_COLS].to_numpy(dtype=np.float32)
            X = stats.apply(X)
            y = group[self.target_cols].to_numpy(dtype=np.float32)
            for end in range(LOOKBACK_STEPS - 1, len(group)):
                target = y[end]
                if np.isnan(target).any():
                    continue
                samples.append({
                    "seq": X[end - LOOKBACK_STEPS + 1 : end + 1],
                    "line_id": int(line_id),
                    "y": target,
                })
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        s = self._samples[idx]
        return (
            torch.from_numpy(s["seq"].copy()),
            torch.tensor(s["line_id"], dtype=torch.long),
            torch.from_numpy(s["y"].copy()),
        )
