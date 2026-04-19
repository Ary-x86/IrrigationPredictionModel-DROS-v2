"""Phase 0 honesty check: permutation importance on the legacy Preite MLP.

Loads models/mlp_irrigation_model.pkl (bundle), scores accuracy on a held-out
split of data/processed_dataset.csv, then shuffles each feature on the test
split and measures accuracy drop. Writes reports/leakage_audit_phase0.md.

Expected finding: Soil Moisture + Weather Forecast Rainfall dominate the
permutation importance, consistent with the label being derived from those
two features in src/03_soil_capacity_calculator.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

SEED = 42


def _load_bundle(path: Path) -> tuple[object, list[str]]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise SystemExit(f"Bundle at {path} missing keys 'model'/'feature_columns'.")
    return bundle["model"], bundle["feature_columns"]


def _split(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy()
    y = df["Irrigation_Decision"].astype(int).copy()
    return train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)


def permutation_accuracy_drop(model, X_test: pd.DataFrame, y_test: pd.Series, repeats: int = 5) -> list[tuple[str, float]]:
    rng = np.random.default_rng(SEED)
    base_acc = accuracy_score(y_test, model.predict(X_test))
    rows = []
    for col in X_test.columns:
        drops = []
        for _ in range(repeats):
            Xp = X_test.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base_acc - accuracy_score(y_test, model.predict(Xp)))
        rows.append((col, float(np.mean(drops))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows, base_acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(MODELS / "mlp_irrigation_model.pkl"))
    p.add_argument("--data", default=str(DATA / "processed_dataset.csv"))
    args = p.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)

    model, feature_cols = _load_bundle(Path(args.model))
    df = pd.read_csv(args.data)
    X_train, X_test, y_train, y_test = _split(df, feature_cols)
    X_test = X_test[feature_cols]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    cr = classification_report(y_test, y_pred, labels=[0, 1, 2, 3],
                               target_names=["OFF", "ON", "NoAdj", "Alert"], zero_division=0)

    perm_rows, base_acc = permutation_accuracy_drop(model, X_test, y_test)

    # Single-feature probe: retrain a shallow decision tree on just Soil Moisture + Rainfall,
    # to show the label is recoverable from those two alone.
    from sklearn.tree import DecisionTreeClassifier
    two = ["Soil Moisture [RH%]", "Weather Forecast Rainfall [mm]"]
    dt = DecisionTreeClassifier(max_depth=4, random_state=SEED)
    dt.fit(X_train[two], y_train)
    two_feat_acc = accuracy_score(y_test, dt.predict(X_test[two]))

    payload = {
        "bundle_path": str(args.model),
        "dataset_path": str(args.data),
        "feature_columns": feature_cols,
        "test_accuracy": float(acc),
        "permutation_base_accuracy": float(base_acc),
        "permutation_drop_by_feature": perm_rows,
        "two_feature_tree_accuracy": float(two_feat_acc),
        "two_features": two,
    }
    (REPORTS / "leakage_audit_phase0.json").write_text(json.dumps(payload, indent=2, default=float))

    md = [
        "# Phase 0 honesty check — legacy Preite MLP leakage audit",
        "",
        f"Bundle: `{args.model}`",
        f"Data:   `{args.data}`",
        "",
        "## Headline",
        "",
        f"- MLP test accuracy (70/30 stratified split, seed 42): **{acc:.4f}**",
        f"- Shallow (max_depth=4) decision tree on **only** `Soil Moisture [RH%]` + "
        f"`Weather Forecast Rainfall [mm]`: **{two_feat_acc:.4f}**",
        "",
        "If the two-feature tree matches the MLP, the label is recoverable from the two "
        "features the labeling rule uses. That is the definition of label leakage.",
        "",
        "## Permutation importance (accuracy drop on held-out 30%, 5 shuffles, mean)",
        "",
        "| feature | accuracy_drop |",
        "| --- | ---: |",
    ]
    for name, drop in perm_rows:
        md.append(f"| {name} | {drop:+.4f} |")

    md += [
        "",
        "## Classification report",
        "",
        "```",
        cr.strip(),
        "```",
        "",
        "## Confusion matrix (rows=true, cols=pred; labels [OFF, ON, NoAdj, Alert])",
        "",
        "```",
        str(cm),
        "```",
        "",
        "## Why this matters",
        "",
        "`src/03_soil_capacity_calculator.py` computes `Irrigation_Decision` with "
        "explicit threshold rules on `Soil Moisture [RH%]` and "
        "`Weather Forecast Rainfall [mm]`. Those two columns are then fed in as features "
        "for training in `src/04_train_neural_network.py`. The MLP is re-learning the "
        "threshold rule, not discovering an irrigation policy. Headline accuracy and any "
        "downstream Monte Carlo water-savings numbers are therefore tautological.",
        "",
        "This motivates the Track B rebuild: reframe the task as future-VWC regression, "
        "drop threshold-derived classification labels entirely.",
    ]

    (REPORTS / "leakage_audit_phase0.md").write_text("\n".join(md))
    print(f"Wrote {REPORTS / 'leakage_audit_phase0.md'}")
    print(f"MLP accuracy: {acc:.4f}  |  two-feature tree: {two_feat_acc:.4f}")


if __name__ == "__main__":
    main()
