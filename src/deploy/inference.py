"""Farmer-facing single-shot inference: features → JSON decision.

Emits:
    {action, volume_mm, reason_top3_features, confidence,
     predicted_vwc_3h, mad_threshold, growth_stage}

Pipeline:
    features (dict or 1-row DataFrame)
      → XGBoost h=3h forecaster (Track A; small, deterministic, shippable)
      → MAD rule policy
      → tree-SHAP top-3 drivers as rationale

Why Track A here: ships standalone with no torch dependency, and SHAP over
XGBoost is the same explainer used in Wagan 2025 for farmer-facing rationale.
Track B LSTM is for research/benchmarking; not the production path yet.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from experiments.gbdt.config import EXP_MODELS
from experiments.gbdt.temporal_split import load_features, split_by_time
from experiments.gbdt.train_xgboost import FEATURE_COLS
from src.config import ACTION_VOLUME_MM, FIELD_CAPACITY_PCT, WILTING_POINT_PCT
from src.policy.policy_rule import mad_threshold

MODEL_PATH = EXP_MODELS / "xgb_h3h.json"


@dataclass
class Decision:
    action: str
    volume_mm: float
    predicted_vwc_3h: float
    mad_threshold: float
    growth_stage: str
    reason_top3_features: list[dict]
    confidence: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=float)


class Forecaster:
    def __init__(self, model_path=MODEL_PATH):
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))
        self.explainer = shap.TreeExplainer(self.booster)

    def predict(self, row: pd.Series) -> tuple[float, np.ndarray]:
        x = row[FEATURE_COLS].to_numpy(dtype=np.float32).reshape(1, -1)
        pred = float(self.booster.predict(xgb.DMatrix(x, feature_names=FEATURE_COLS))[0])
        sv = self.explainer.shap_values(x)[0]
        return pred, sv


def _rule_action(vwc_hat: float, stage: str) -> tuple[str, float, float]:
    hi = mad_threshold(stage)
    lo = hi - 2.0
    if vwc_hat < lo:
        return "ON_HIGH", ACTION_VOLUME_MM["ON_HIGH"], hi
    if vwc_hat < hi:
        return "ON_LOW", ACTION_VOLUME_MM["ON_LOW"], hi
    return "OFF", 0.0, hi


def _confidence(vwc_hat: float, mad_hi: float) -> float:
    taw = FIELD_CAPACITY_PCT - WILTING_POINT_PCT
    return float(min(abs(vwc_hat - mad_hi) / taw, 1.0))


def _top3(sv: np.ndarray, row: pd.Series) -> list[dict]:
    order = np.argsort(-np.abs(sv))[:3]
    out = []
    for j in order:
        feat = FEATURE_COLS[j]
        out.append({
            "feature": feat,
            "value": float(row[feat]),
            "shap": float(sv[j]),
            "direction": "drier" if sv[j] < 0 else "wetter",
        })
    return out


def decide(row: pd.Series, forecaster: Forecaster | None = None) -> Decision:
    forecaster = forecaster or Forecaster()
    pred, sv = forecaster.predict(row)
    stage = str(row.get("growth_stage", "mid"))
    action, vol, mad_hi = _rule_action(pred, stage)
    return Decision(
        action=action,
        volume_mm=vol,
        predicted_vwc_3h=pred,
        mad_threshold=mad_hi,
        growth_stage=stage,
        reason_top3_features=_top3(sv, row),
        confidence=_confidence(pred, mad_hi),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--row", type=int, default=-1, help="test-split row index; -1 = last row")
    args = p.parse_args()

    df = load_features()
    parts = split_by_time(df)
    test = parts["test"].reset_index(drop=True)
    if test.empty:
        raise SystemExit("Empty test split.")
    idx = args.row if args.row >= 0 else len(test) - 1
    row = test.iloc[idx]
    d = decide(row)
    print(d.to_json())


if __name__ == "__main__":
    main()
