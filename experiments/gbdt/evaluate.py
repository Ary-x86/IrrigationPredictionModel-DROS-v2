"""Evaluate XGB + LGB + baselines on val/test and simulate the policy layer.

Writes experiments/gbdt/reports/report.md — the Track A deliverable.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from experiments.gbdt.baselines import (
    climatology_forecast,
    persistence_forecast,
    water_balance_forecast,
)
from experiments.gbdt.config import (
    ACTION_VOLUME_MM,
    EXP_MODELS,
    EXP_REPORTS,
    HORIZON_LABELS,
    IRRIFRAME_SEASON_MM,
)
from experiments.gbdt.policy_rule import decide
from experiments.gbdt.temporal_split import load_features, split_by_time
from experiments.gbdt.train_xgboost import FEATURE_COLS, _mae, _nse, _rmse


def _xgb_preds(split_df: pd.DataFrame) -> dict[str, np.ndarray]:
    X = split_df[FEATURE_COLS].values
    out = {}
    for label in HORIZON_LABELS:
        booster = xgb.Booster()
        booster.load_model(str(EXP_MODELS / f"xgb_h{label}.json"))
        d = xgb.DMatrix(X, feature_names=FEATURE_COLS)
        out[label] = booster.predict(d)
    return out


def _lgb_preds(split_df: pd.DataFrame) -> dict[str, np.ndarray]:
    X = split_df[FEATURE_COLS].values
    out = {}
    for label in HORIZON_LABELS:
        booster = lgb.Booster(model_file=str(EXP_MODELS / f"lgb_h{label}.txt"))
        out[label] = booster.predict(X)
    return out


def _metrics_table(split_df: pd.DataFrame, preds: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for label in HORIZON_LABELS:
        y = split_df[f"y_vwc_h{label}"].values
        mask = ~np.isnan(y)
        y_m, p_m = y[mask], preds[label][mask]
        if len(y_m) == 0:
            rows.append({"h": label, "rmse": float("nan"), "mae": float("nan"), "nse": float("nan"), "n": 0})
            continue
        rows.append({
            "h": label,
            "rmse": _rmse(y_m, p_m),
            "mae": _mae(y_m, p_m),
            "nse": _nse(y_m, p_m),
            "n": int(mask.sum()),
        })
    return rows


def _policy_backtest(split_df: pd.DataFrame, pred_3h: np.ndarray) -> dict:
    decisions = decide(pred_3h, split_df["growth_stage"])
    total_days = (split_df["datetime"].max() - split_df["datetime"].min()).total_seconds() / 86400.0
    total_days = max(total_days, 1e-6)

    # each "decision" is a 10-min step; one volume event per step is unrealistic,
    # so collapse to an hourly decision: only count the first decision in each hour per line
    hourly = split_df.assign(_action=decisions["action"].values,
                             _volume=decisions["volume_mm"].values)
    hourly["_hour"] = hourly["datetime"].dt.floor("h")
    first_per_hour = hourly.drop_duplicates(subset=["line", "_hour"], keep="first")
    total_volume_mm = float(first_per_hour["_volume"].sum())
    n_lines = first_per_hour["line"].nunique()
    per_line_mm = total_volume_mm / max(n_lines, 1)
    per_line_mm_per_day = per_line_mm / total_days

    stuard_mm = float(split_df["volume_diff"].fillna(0.0).sum()) / max(n_lines, 1)

    action_counts = first_per_hour["_action"].value_counts().to_dict()
    return {
        "days": round(total_days, 2),
        "n_lines": int(n_lines),
        "policy_mm_per_line": round(per_line_mm, 2),
        "policy_mm_per_line_per_day": round(per_line_mm_per_day, 3),
        "stuard_mm_per_line": round(stuard_mm, 2),
        "irriframe_mm_per_season": IRRIFRAME_SEASON_MM,
        "action_counts_hourly": action_counts,
    }


def run():
    df = load_features()
    parts = split_by_time(df)
    val, test = parts["val"], parts["train"]  # placeholder, override below
    val = parts["val"]
    test = parts["test"]

    xgb_val = _xgb_preds(val)
    xgb_test = _xgb_preds(test)
    lgb_val = _lgb_preds(val)
    lgb_test = _lgb_preds(test)

    per_val = persistence_forecast(val)
    per_test = persistence_forecast(test)
    clim = climatology_forecast(parts["train"], val)
    clim_t = climatology_forecast(parts["train"], test)
    wb_val = water_balance_forecast(val)
    wb_test = water_balance_forecast(test)

    models = {
        "persistence": (per_val, per_test),
        "climatology": (clim, clim_t),
        "water_balance": (wb_val, wb_test),
        "xgboost": ({l: xgb_val[l] for l in HORIZON_LABELS}, {l: xgb_test[l] for l in HORIZON_LABELS}),
        "lightgbm": ({l: lgb_val[l] for l in HORIZON_LABELS}, {l: lgb_test[l] for l in HORIZON_LABELS}),
    }

    # coerce the baseline pred DataFrames into {label: ndarray}
    def _to_map(obj, split_df):
        if isinstance(obj, pd.DataFrame):
            return {l: obj[f"pred_h{l}"].values for l in HORIZON_LABELS}
        return obj

    table = {}
    for name, (v, t) in models.items():
        vmap = _to_map(v, val)
        tmap = _to_map(t, test)
        table[name] = {"val": _metrics_table(val, vmap), "test": _metrics_table(test, tmap)}

    policy = {
        "xgboost_val": _policy_backtest(val, xgb_val["3h"]),
        "xgboost_test": _policy_backtest(test, xgb_test["3h"]),
        "lightgbm_val": _policy_backtest(val, lgb_val["3h"]),
        "lightgbm_test": _policy_backtest(test, lgb_test["3h"]),
    }

    EXP_REPORTS.mkdir(parents=True, exist_ok=True)
    (EXP_REPORTS / "evaluation.json").write_text(
        json.dumps({"metrics": table, "policy": policy}, indent=2, default=float)
    )

    md = [
        "# Track A — GBDT forecaster + rule policy", "",
        "Forecast target: future soil VWC. No threshold-derived label; see plan.",
        "",
        "## Forecast metrics (RMSE | MAE | NSE)", "",
    ]
    for split_name in ("val", "test"):
        md.append(f"### {split_name}")
        md.append("")
        md.append("| model | " + " | ".join(HORIZON_LABELS) + " |")
        md.append("| --- | " + " | ".join(["---:"] * len(HORIZON_LABELS)) + " |")
        for name in ("persistence", "climatology", "water_balance", "xgboost", "lightgbm"):
            row = table[name][split_name]
            cells = []
            for r in row:
                cells.append(f"{r['rmse']:.2f} / {r['mae']:.2f} / {r['nse']:.2f}")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")

    md.append("## Policy backtest (mm of water per line)")
    md.append("")
    md.append("| split/model | days | lines | policy mm/line | policy mm/line/day | Stuard mm/line | action counts |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for key, val_pol in policy.items():
        md.append(
            f"| {key} | {val_pol['days']} | {val_pol['n_lines']} | "
            f"{val_pol['policy_mm_per_line']} | {val_pol['policy_mm_per_line_per_day']} | "
            f"{val_pol['stuard_mm_per_line']} | {val_pol['action_counts_hourly']} |"
        )
    md.append("")
    md.append(f"IRRIFRAME full-season reference: **{IRRIFRAME_SEASON_MM} mm**.")
    md.append("")
    md.append("## Notes")
    md.append("- Backing papers: Hamdaoui 2024 (tree ensembles on scarce tabular); Wagan 2025 (SHAP + XGB).")
    md.append("- This report feeds Phase 6 cross-track comparison vs Track B LSTM.")
    (EXP_REPORTS / "report.md").write_text("\n".join(md))
    print("Wrote", EXP_REPORTS / "report.md")


if __name__ == "__main__":
    run()
