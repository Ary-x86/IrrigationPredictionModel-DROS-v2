"""Run baselines + trained LSTM on val and test. Write reports/baseline_comparison.md.

Baselines: persistence, water-balance bucket.
Model: trained LSTM from models/forecaster_lstm.pt.

For each (model, split, horizon) emit RMSE / MAE / NSE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import DATA_PROCESSED, HORIZON_LABELS, MODELS, REPORTS
from src.eval.metrics import mae, nse, rmse
from src.models.lstm_forecaster import LSTMConfig, LSTMForecaster
from src.models import persistence_baseline, water_balance
from src.splits.temporal_split import split
from src.training.dataset import SEQ_FEATURE_COLS, FeatureStats, VWCSequenceDataset

DATASET = DATA_PROCESSED / "modeling_dataset_v2.parquet"
CHECKPOINT = MODELS / "forecaster_lstm.pt"


def _load_lstm(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LSTMConfig(**ck["config"])
    model = LSTMForecaster(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    stats = FeatureStats(
        mean=np.array(ck["stats_mean"], dtype=np.float32),
        std=np.array(ck["stats_std"], dtype=np.float32),
        columns=ck["feature_cols"],
    )
    return model, stats


def _lstm_preds(split_df: pd.DataFrame, model, stats) -> dict[str, np.ndarray]:
    """Return per-row forecasts aligned to split_df index. Rows without a full
    144-step lookback return NaN."""
    ds = VWCSequenceDataset(split_df, stats)
    preds = np.full((len(split_df), len(HORIZON_LABELS)), np.nan, dtype=np.float32)

    # rebuild the same index mapping the dataset used, so we can place predictions
    # back onto the original rows.
    idx = 0
    placements: list[int] = []
    for line_id, group in split_df.groupby("line_id", sort=False):
        group = group.sort_values("datetime").reset_index()
        target_cols = [f"y_vwc_h{h}" for h in HORIZON_LABELS]
        y = group[target_cols].to_numpy(dtype=np.float32)
        for end in range(143, len(group)):
            if np.isnan(y[end]).any():
                continue
            placements.append(int(group.loc[end, "index"]))

    if placements:
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        batched = []
        with torch.no_grad():
            for seq, line_id, _ in loader:
                batched.append(model(seq, line_id).numpy())
        out = np.concatenate(batched, axis=0)
        for row_idx, p in zip(placements, out):
            preds[row_idx] = p

    return {label: preds[:, i] for i, label in enumerate(HORIZON_LABELS)}


def _table(split_df: pd.DataFrame, pred_map: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for label in HORIZON_LABELS:
        y = split_df[f"y_vwc_h{label}"].to_numpy()
        p = pred_map[label]
        rows.append({"h": label,
                     "rmse": rmse(y, p), "mae": mae(y, p), "nse": nse(y, p),
                     "n": int((~np.isnan(y) & ~np.isnan(p)).sum())})
    return rows


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATASET)
    parts = split(df)
    val = parts["val"].reset_index(drop=True)
    test = parts["test"].reset_index(drop=True)

    model, stats = _load_lstm(CHECKPOINT)

    tables = {"val": {}, "test": {}}
    for split_name, sdf in (("val", val), ("test", test)):
        tables[split_name]["persistence"] = _table(sdf, persistence_baseline.predict(sdf))
        tables[split_name]["water_balance"] = _table(sdf, water_balance.predict(sdf))
        tables[split_name]["lstm"] = _table(sdf, _lstm_preds(sdf, model, stats))

    (REPORTS / "baseline_comparison.json").write_text(json.dumps(tables, indent=2, default=float))

    md = [
        "# Track B baseline comparison (LSTM vs physics/persistence)", "",
        "Dataset: `data/processed/modeling_dataset_v2.parquet`.",
        "Model: `models/forecaster_lstm.pt` (Track B LSTM, Stuard fine-tune only; no ERA5 pretraining yet).",
        "",
    ]
    for split_name in ("val", "test"):
        md.append(f"## {split_name} — RMSE / MAE / NSE")
        md.append("")
        md.append("| model | " + " | ".join(HORIZON_LABELS) + " |")
        md.append("| --- | " + " | ".join(["---:"] * len(HORIZON_LABELS)) + " |")
        for name in ("persistence", "water_balance", "lstm"):
            row = tables[split_name][name]
            cells = [f"{r['rmse']:.2f} / {r['mae']:.2f} / {r['nse']:.2f}" for r in row]
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")
    md += [
        "## Notes",
        "- Acceptance per plan: LSTM must beat persistence at h=1h and beat water-balance at h=6h on val.",
        "- Phase 6 will run this table alongside Track A GBDT numbers in `reports/track_comparison.md`.",
    ]
    (REPORTS / "baseline_comparison.md").write_text("\n".join(md))
    print(f"Wrote {REPORTS / 'baseline_comparison.md'}")


if __name__ == "__main__":
    main()
