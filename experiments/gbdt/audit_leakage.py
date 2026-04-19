"""Leakage audit for Track A.

Three checks on the trained XGBoost h=1h model:
  1. Shuffle-label sanity: retrain on shuffled y, must be worse than persistence.
  2. Permutation importance on val: drop in RMSE per feature.
  3. Feature ablation: drop vwc_20cm + lags, retrain, show model cannot coast.

Writes experiments/gbdt/reports/leakage_audit.md.
"""
from __future__ import annotations

import json
import time

import numpy as np
import xgboost as xgb

from experiments.gbdt.config import EXP_REPORTS
from experiments.gbdt.temporal_split import load_features, split_by_time
from experiments.gbdt.train_xgboost import FEATURE_COLS, NUM_BOOST_ROUND, PARAMS, _rmse

SEED = 0


def _train(X_tr, y_tr, X_vl, y_vl, num_round=NUM_BOOST_ROUND, features=None):
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
    dval = xgb.DMatrix(X_vl, label=y_vl, feature_names=features)
    booster = xgb.train(
        PARAMS, dtrain, num_boost_round=num_round,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return booster


def shuffle_label_test(train_df, val_df, target: str) -> dict:
    rng = np.random.default_rng(SEED)
    y_tr = train_df[target].values.copy()
    rng.shuffle(y_tr)
    X_tr = train_df[FEATURE_COLS].values
    X_vl = val_df[FEATURE_COLS].values
    y_vl = val_df[target].values

    booster = _train(X_tr, y_tr, X_vl, y_vl, features=FEATURE_COLS)
    dval = xgb.DMatrix(X_vl, feature_names=FEATURE_COLS)
    preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    shuffled_rmse = _rmse(y_vl, preds)

    persistence = val_df["vwc_20cm"].values
    persistence_rmse = _rmse(y_vl, persistence)
    return {
        "shuffled_val_rmse": shuffled_rmse,
        "persistence_val_rmse": persistence_rmse,
        "pass": shuffled_rmse >= persistence_rmse,
    }


def permutation_importance(train_df, val_df, target: str) -> dict:
    X_tr = train_df[FEATURE_COLS].values
    y_tr = train_df[target].values
    X_vl = val_df[FEATURE_COLS].values.copy()
    y_vl = val_df[target].values

    booster = _train(X_tr, y_tr, X_vl, y_vl, features=FEATURE_COLS)
    dval = xgb.DMatrix(X_vl, feature_names=FEATURE_COLS)
    base_rmse = _rmse(y_vl, booster.predict(dval, iteration_range=(0, booster.best_iteration + 1)))

    rng = np.random.default_rng(SEED)
    rows = []
    for i, name in enumerate(FEATURE_COLS):
        Xp = X_vl.copy()
        rng.shuffle(Xp[:, i])
        dperm = xgb.DMatrix(Xp, feature_names=FEATURE_COLS)
        rmse = _rmse(y_vl, booster.predict(dperm, iteration_range=(0, booster.best_iteration + 1)))
        rows.append((name, rmse - base_rmse))
    rows.sort(key=lambda x: x[1], reverse=True)
    return {"base_rmse": base_rmse, "importance": rows}


def ablation_test(train_df, val_df, target: str) -> dict:
    drop = {c for c in FEATURE_COLS if c.startswith("vwc_20cm")}
    feats = [c for c in FEATURE_COLS if c not in drop]
    X_tr = train_df[feats].values
    y_tr = train_df[target].values
    X_vl = val_df[feats].values
    y_vl = val_df[target].values

    booster = _train(X_tr, y_tr, X_vl, y_vl, features=feats)
    dval = xgb.DMatrix(X_vl, feature_names=feats)
    preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    ablated_rmse = _rmse(y_vl, preds)

    dtrain = xgb.DMatrix(train_df[FEATURE_COLS].values, label=y_tr, feature_names=FEATURE_COLS)
    dval_full = xgb.DMatrix(val_df[FEATURE_COLS].values, label=y_vl, feature_names=FEATURE_COLS)
    booster_full = xgb.train(
        PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval_full, "val")],
        early_stopping_rounds=50, verbose_eval=False,
    )
    full_rmse = _rmse(y_vl, booster_full.predict(dval_full, iteration_range=(0, booster_full.best_iteration + 1)))
    return {
        "dropped_features": sorted(drop),
        "ablated_val_rmse": ablated_rmse,
        "full_val_rmse": full_rmse,
        "degradation_rmse": ablated_rmse - full_rmse,
    }


def run(target: str = "y_vwc_h1h") -> dict:
    EXP_REPORTS.mkdir(parents=True, exist_ok=True)
    df = load_features()
    parts = split_by_time(df)
    tr = parts["train"].dropna(subset=[target])
    vl = parts["val"].dropna(subset=[target])

    t0 = time.time()
    shuffle = shuffle_label_test(tr, vl, target)
    perm = permutation_importance(tr, vl, target)
    abl = ablation_test(tr, vl, target)
    dt = time.time() - t0

    report = {"target": target, "elapsed_seconds": round(dt, 1),
              "shuffle_label": shuffle, "ablation": abl,
              "permutation_importance": perm["importance"],
              "permutation_base_rmse": perm["base_rmse"]}

    (EXP_REPORTS / "leakage_audit.json").write_text(json.dumps(report, indent=2, default=float))

    md = ["# Track A leakage audit", "",
          f"Target: `{target}`  ·  elapsed {dt:.1f}s", "",
          "## Shuffle-label sanity",
          f"- shuffled val RMSE: **{shuffle['shuffled_val_rmse']:.3f}**",
          f"- persistence val RMSE: **{shuffle['persistence_val_rmse']:.3f}**",
          f"- pass (shuffled >= persistence): **{shuffle['pass']}**", "",
          "## Feature ablation (drop vwc_20cm + lags)",
          f"- full-feature val RMSE: **{abl['full_val_rmse']:.3f}**",
          f"- ablated val RMSE: **{abl['ablated_val_rmse']:.3f}**",
          f"- degradation: **{abl['degradation_rmse']:+.3f}**",
          f"- dropped: {', '.join(abl['dropped_features'])}", "",
          "## Permutation importance (val RMSE - base)", ""]
    md.append("| feature | delta_rmse |")
    md.append("| --- | ---: |")
    for name, delta in perm["importance"]:
        md.append(f"| {name} | {delta:+.4f} |")
    (EXP_REPORTS / "leakage_audit.md").write_text("\n".join(md))
    print("Wrote", EXP_REPORTS / "leakage_audit.md")
    return report


if __name__ == "__main__":
    run()
