"""Train one XGBoost regressor per horizon for Track A.

Backing: Hamdaoui 2024 flags tree ensembles (RF/XGB/CatBoost) as strong on
data-scarce tabular tasks; Wagan 2025 applies SHAP directly to XGBoost for
irrigation interpretability.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from experiments.gbdt.config import EXP_MODELS, EXP_REPORTS, HORIZON_LABELS
from experiments.gbdt.temporal_split import load_features, split_by_time

FEATURE_COLS = [
    "vwc_20cm",
    "vwc_20cm_lag_6s", "vwc_20cm_lag_18s", "vwc_20cm_lag_36s",
    "vwc_20cm_lag_72s", "vwc_20cm_lag_144s",
    "vwc_20cm_roll_mean_3h", "vwc_20cm_roll_std_6h", "vwc_20cm_deriv_1h",
    "soil_temp_c", "air_temp_c", "rh_pct", "vpd_kpa",
    "rain_mm_h", "rain_mm_1h_sum", "rain_mm_6h_sum", "rain_mm_24h_sum",
    "et0_open_meteo_mm_h", "et0_hargreaves_mm_day", "etc_mm_h",
    "ra_mj_m2_day", "daylength_h",
    "gdd_cum", "kc_dynamic",
    "sin_doy", "cos_doy", "sin_hour", "cos_hour",
    "hours_since_last_irrigation",
    "line_id",
]

PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
)
NUM_BOOST_ROUND = 1000
EARLY_STOPPING = 50


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def train_all_horizons() -> dict:
    df = load_features()
    parts = split_by_time(df)
    train_df, val_df, test_df = parts["train"], parts["val"], parts["test"]

    EXP_MODELS.mkdir(parents=True, exist_ok=True)
    EXP_REPORTS.mkdir(parents=True, exist_ok=True)

    report = {"feature_cols": FEATURE_COLS, "params": PARAMS, "horizons": {}}
    t_total = time.time()

    for label in HORIZON_LABELS:
        target = f"y_vwc_h{label}"
        tr = train_df.dropna(subset=[target])
        vl = val_df.dropna(subset=[target])
        te = test_df.dropna(subset=[target])

        X_tr, y_tr = tr[FEATURE_COLS].values, tr[target].values
        X_vl, y_vl = vl[FEATURE_COLS].values, vl[target].values
        X_te, y_te = te[FEATURE_COLS].values, te[target].values

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_COLS)
        dval = xgb.DMatrix(X_vl, label=y_vl, feature_names=FEATURE_COLS)
        dtest = xgb.DMatrix(X_te, label=y_te, feature_names=FEATURE_COLS)

        t0 = time.time()
        booster = xgb.train(
            PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False,
        )
        dt = time.time() - t0

        best_iter = booster.best_iteration
        p_tr = booster.predict(dtrain, iteration_range=(0, best_iter + 1))
        p_vl = booster.predict(dval, iteration_range=(0, best_iter + 1))
        p_te = booster.predict(dtest, iteration_range=(0, best_iter + 1))

        model_path = EXP_MODELS / f"xgb_h{label}.json"
        booster.save_model(str(model_path))

        report["horizons"][label] = {
            "best_iter": int(best_iter),
            "train_seconds": round(dt, 2),
            "train_rmse": _rmse(y_tr, p_tr),
            "val_rmse": _rmse(y_vl, p_vl),
            "test_rmse": _rmse(y_te, p_te),
            "val_mae": _mae(y_vl, p_vl),
            "test_mae": _mae(y_te, p_te),
            "val_nse": _nse(y_vl, p_vl),
            "test_nse": _nse(y_te, p_te),
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_vl)),
            "n_test": int(len(y_te)),
            "model_path": str(model_path),
        }
        print(
            f"[xgb h={label}] best_iter={best_iter}  "
            f"val_rmse={report['horizons'][label]['val_rmse']:.3f}  "
            f"test_rmse={report['horizons'][label]['test_rmse']:.3f}  "
            f"test_nse={report['horizons'][label]['test_nse']:.3f}  "
            f"({dt:.1f}s)"
        )

    report["total_seconds"] = round(time.time() - t_total, 2)
    out_json = EXP_REPORTS / "xgb_training_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    return report


if __name__ == "__main__":
    train_all_horizons()
