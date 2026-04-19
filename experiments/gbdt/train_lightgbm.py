"""Train one LightGBM regressor per horizon for Track A.

Mirror of train_xgboost.py with the same feature set, splits, and metrics so
the two GBDT variants are directly comparable at evaluate time.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np

from experiments.gbdt.config import EXP_MODELS, EXP_REPORTS, HORIZON_LABELS
from experiments.gbdt.temporal_split import load_features, split_by_time
from experiments.gbdt.train_xgboost import FEATURE_COLS, _mae, _nse, _rmse

PARAMS = dict(
    objective="regression",
    metric="rmse",
    num_leaves=63,
    max_depth=-1,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    min_data_in_leaf=20,
    lambda_l2=1.0,
    verbose=-1,
)
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 50


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

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
        dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain, feature_name=FEATURE_COLS)

        t0 = time.time()
        booster = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
        )
        dt = time.time() - t0

        best_iter = booster.best_iteration
        p_tr = booster.predict(X_tr, num_iteration=best_iter)
        p_vl = booster.predict(X_vl, num_iteration=best_iter)
        p_te = booster.predict(X_te, num_iteration=best_iter)

        model_path = EXP_MODELS / f"lgb_h{label}.txt"
        booster.save_model(str(model_path), num_iteration=best_iter)

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
            f"[lgb h={label}] best_iter={best_iter}  "
            f"val_rmse={report['horizons'][label]['val_rmse']:.3f}  "
            f"test_rmse={report['horizons'][label]['test_rmse']:.3f}  "
            f"test_nse={report['horizons'][label]['test_nse']:.3f}  "
            f"({dt:.1f}s)"
        )

    report["total_seconds"] = round(time.time() - t_total, 2)
    out_json = EXP_REPORTS / "lgb_training_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    return report


if __name__ == "__main__":
    train_all_horizons()
