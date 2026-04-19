"""Backtest all three policy layers (FAO-56, rule, MPC) on val + test.

Writes reports/policy_backtest.md. Uses the trained LSTM's forecasts as the
input to the rule and MPC policies. FAO-56 runs on the current-step features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import DATA_PROCESSED, HORIZON_LABELS, IRRIFRAME_SEASON_MM, MODELS, REPORTS
from src.models.lstm_forecaster import LSTMConfig, LSTMForecaster
from src.policy import policy_fao56, policy_mpc, policy_rule
from src.splits.temporal_split import split
from src.training.dataset import FeatureStats, VWCSequenceDataset

DATASET = DATA_PROCESSED / "modeling_dataset_v2.parquet"
CHECKPOINT = MODELS / "forecaster_lstm.pt"


def _load_lstm():
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
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


def _lstm_predict_per_row(split_df: pd.DataFrame, model, stats) -> dict[str, np.ndarray]:
    preds = np.full((len(split_df), len(HORIZON_LABELS)), np.nan, dtype=np.float32)
    placements: list[int] = []
    tensors: list[torch.Tensor] = []
    line_ids: list[int] = []
    for line_id, group in split_df.groupby("line_id", sort=False):
        group = group.sort_values("datetime").reset_index()
        X = group[stats.columns].to_numpy(dtype=np.float32)
        X = (X - stats.mean) / stats.std
        for end in range(143, len(group)):
            placements.append(int(group.loc[end, "index"]))
            tensors.append(torch.from_numpy(X[end - 143: end + 1]))
            line_ids.append(int(line_id))
    if not placements:
        return {label: preds[:, i] for i, label in enumerate(HORIZON_LABELS)}
    with torch.no_grad():
        batch_seq = torch.stack(tensors)
        batch_id = torch.tensor(line_ids, dtype=torch.long)
        out = model(batch_seq, batch_id).numpy()
    for row_idx, p in zip(placements, out):
        preds[row_idx] = p
    return {label: preds[:, i] for i, label in enumerate(HORIZON_LABELS)}


def _policy_summary(split_df: pd.DataFrame, actions: pd.DataFrame) -> dict:
    joined = split_df.reset_index(drop=True).assign(
        _action=actions["action"].values,
        _volume=actions["volume_mm"].values,
    )
    joined["_hour"] = joined["datetime"].dt.floor("h")
    hourly = joined.drop_duplicates(subset=["line", "_hour"], keep="first")
    total_mm = float(hourly["_volume"].sum())
    n_lines = int(hourly["line"].nunique())
    days = max((joined["datetime"].max() - joined["datetime"].min()).total_seconds() / 86400.0, 1e-6)
    return {
        "days": round(days, 2),
        "n_lines": n_lines,
        "policy_mm_per_line": round(total_mm / max(n_lines, 1), 2),
        "policy_mm_per_line_per_day": round((total_mm / max(n_lines, 1)) / days, 3),
        "stuard_mm_per_line": round(float(split_df["volume_diff"].fillna(0.0).sum()) / max(n_lines, 1), 2),
        "action_counts": hourly["_action"].value_counts().to_dict(),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATASET)
    parts = split(df)
    val, test = parts["val"].reset_index(drop=True), parts["test"].reset_index(drop=True)

    model, stats = _load_lstm()

    results: dict[str, dict] = {}
    for name, sdf in (("val", val), ("test", test)):
        fmap = _lstm_predict_per_row(sdf, model, stats)
        # fallback to persistence for rows without enough lookback (policies need a value)
        persistence = sdf["vwc_20cm"].to_numpy(dtype=np.float32)
        for k in fmap:
            fmap[k] = np.where(np.isnan(fmap[k]), persistence, fmap[k])

        fao_actions = policy_fao56.decide(sdf)
        rule_actions = policy_rule.decide(fmap["3h"], sdf["growth_stage"])
        mpc_actions = policy_mpc.decide(fmap, sdf["growth_stage"])

        results[name] = {
            "fao56": _policy_summary(sdf, fao_actions),
            "rule_lstm_3h": _policy_summary(sdf, rule_actions),
            "mpc_lstm_6h": _policy_summary(sdf, mpc_actions),
        }

    (REPORTS / "policy_backtest.json").write_text(json.dumps(results, indent=2, default=float))

    md = [
        "# Track B policy backtest",
        "",
        "Policies:",
        "- `fao56`: Allen FAO-56 ETc-threshold scheduler (current-step only).",
        "- `rule_lstm_3h`: rule policy on the LSTM's t+3h forecast.",
        "- `mpc_lstm_6h`: MPC optimizing cost over the LSTM's 1h/3h/6h trajectory.",
        "",
        f"IRRIFRAME reference (full 2023 season): **{IRRIFRAME_SEASON_MM} mm** per plot.",
        "",
    ]
    for split_name in ("val", "test"):
        md.append(f"## {split_name}")
        md.append("")
        md.append("| policy | days | lines | mm/line | mm/line/day | Stuard mm/line | actions |")
        md.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for name in ("fao56", "rule_lstm_3h", "mpc_lstm_6h"):
            s = results[split_name][name]
            md.append(
                f"| {name} | {s['days']} | {s['n_lines']} | "
                f"{s['policy_mm_per_line']} | {s['policy_mm_per_line_per_day']} | "
                f"{s['stuard_mm_per_line']} | {s['action_counts']} |"
            )
        md.append("")
    md += [
        "## Caveats",
        "- `stuard_mm_per_line` reads `volume_diff` from the raw sensor feed; on this feed it likely reflects cumulative-counter semantics, not per-step deltas. Treat as a placeholder column until the water-meter schema is validated.",
        "- MPC wetting bump uses a fixed `WETTING_VWC_PER_MM = 0.5` (20cm root depth). Phase 6 will couple the bump to the forecaster's residual-under-action.",
    ]
    (REPORTS / "policy_backtest.md").write_text("\n".join(md))
    print(f"Wrote {REPORTS / 'policy_backtest.md'}")


if __name__ == "__main__":
    main()
