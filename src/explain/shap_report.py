"""Per-decision SHAP explanations over the Track A XGBoost forecaster.

Wagan 2025 shows SHAP over XGBoost gives farmer-readable rationales for each
irrigation decision. We explain the h=3h forecaster (the one the rule policy
reads) and aggregate top-3 drivers per sampled decision.

Why Track A and not the LSTM: DeepExplainer on the PyTorch LSTM is ~100x
slower per sample and Wagan's paper specifically uses tree-SHAP. LSTM
attributions are deferred to Phase 6b.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from experiments.gbdt.config import EXP_MODELS
from experiments.gbdt.temporal_split import load_features, split_by_time
from experiments.gbdt.train_xgboost import FEATURE_COLS
from src.config import REPORTS

MODEL_PATH = EXP_MODELS / "xgb_h3h.json"
OUT_DIR = REPORTS / "shap_examples"


def _load_model() -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster


def _sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if len(df) <= n:
        return df.copy()
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)].copy()


def run(n_samples: int = 20, seed: int = 0) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    parts = split_by_time(df)
    test = parts["test"].reset_index(drop=True)
    if test.empty:
        raise SystemExit("Empty test split.")

    sample = _sample_rows(test, n_samples, seed)
    X = sample[FEATURE_COLS].to_numpy(dtype=np.float32)

    booster = _load_model()
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)

    base = float(explainer.expected_value)
    records = []
    for i, row in enumerate(X):
        order = np.argsort(-np.abs(sv[i]))[:3]
        drivers = [
            {
                "feature": FEATURE_COLS[j],
                "value": float(row[j]),
                "shap": float(sv[i, j]),
            }
            for j in order
        ]
        records.append({
            "datetime": str(sample["datetime"].iloc[i]),
            "line_id": int(sample["line_id"].iloc[i]),
            "predicted_vwc_3h": float(base + sv[i].sum()),
            "top3_drivers": drivers,
        })

    agg = pd.DataFrame(
        np.abs(sv).mean(axis=0),
        index=FEATURE_COLS,
        columns=["mean_abs_shap"],
    ).sort_values("mean_abs_shap", ascending=False)

    (OUT_DIR / "decisions.json").write_text(json.dumps(records, indent=2))
    (OUT_DIR / "feature_importance.csv").write_text(agg.to_csv())

    md = [
        "# SHAP report (Track A XGBoost h=3h)",
        "",
        f"Samples: **{len(records)}** from test split. Model: `models/xgb_h3h.json`.",
        "Backing: Wagan 2025 — tree-SHAP for per-decision irrigation rationale.",
        "",
        "## Mean |SHAP| across sampled decisions (top 10)",
        "",
        "| feature | mean |SHAP| |",
        "| --- | ---: |",
    ]
    for feat, val in agg.head(10).itertuples():
        md.append(f"| {feat} | {val:.4f} |")
    md += [
        "",
        "## Per-decision top-3 drivers (first 5)",
        "",
        "| datetime | line | pred VWC 3h | top feature | 2nd | 3rd |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for r in records[:5]:
        d = r["top3_drivers"]
        md.append(
            f"| {r['datetime']} | {r['line_id']} | "
            f"{r['predicted_vwc_3h']:.2f} | "
            f"{d[0]['feature']} ({d[0]['shap']:+.2f}) | "
            f"{d[1]['feature']} ({d[1]['shap']:+.2f}) | "
            f"{d[2]['feature']} ({d[2]['shap']:+.2f}) |"
        )
    md += [
        "",
        "## Notes",
        "- Base value (expected prediction on train) is added to the SHAP sum per row.",
        "- `top3_drivers` feeds deployment's `reason_top3_features`.",
        "- LSTM DeepExplainer is deferred (~100x slower, not needed for farmer UI).",
    ]
    (OUT_DIR.parent / "shap_report.md").write_text("\n".join(md))
    return {"n": len(records), "top_feature": agg.index[0]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    summary = run(args.sample, args.seed)
    print(f"Wrote {REPORTS / 'shap_report.md'} ({summary['n']} decisions, top={summary['top_feature']})")


if __name__ == "__main__":
    main()
