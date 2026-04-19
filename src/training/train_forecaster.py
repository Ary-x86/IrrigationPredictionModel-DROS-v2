"""Train the Track B LSTM forecaster on Stuard data.

Fine-tune-only variant (no ERA5 pretraining yet; see fetch_era5.py docs).
Uses per-horizon Huber loss with weights [1, 0.8, 0.6, 0.4, 0.3].

Outputs:
  models/forecaster_lstm.pt           — {state_dict, stats, config}
  reports/lstm_training_report.json   — per-epoch and final per-horizon metrics
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (
    DATA_PROCESSED, HORIZON_LABELS, HORIZON_LOSS_WEIGHTS, HUBER_DELTA,
    MODELS, REPORTS,
)
from src.models.lstm_forecaster import LSTMConfig, LSTMForecaster
from src.splits.temporal_split import split
from src.training.dataset import SEQ_FEATURE_COLS, VWCSequenceDataset, compute_stats

SEED = 0
BATCH = 128
EPOCHS = 80
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
DATASET = DATA_PROCESSED / "modeling_dataset_v2.parquet"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per = torch.nn.functional.huber_loss(pred, target, delta=HUBER_DELTA, reduction="none")
    return (per * weights).mean()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def _mae(a, b):  return float(np.mean(np.abs(a - b)))
    def _nse(a, b):
        denom = float(np.sum((a - a.mean()) ** 2))
        return 1.0 - float(np.sum((a - b) ** 2)) / denom if denom > 0 else float("nan")
    out: dict[str, float] = {}
    for i, label in enumerate(HORIZON_LABELS):
        out[f"rmse_{label}"] = _rmse(y_true[:, i], y_pred[:, i])
        out[f"mae_{label}"] = _mae(y_true[:, i], y_pred[:, i])
        out[f"nse_{label}"] = _nse(y_true[:, i], y_pred[:, i])
    return out


def run_epoch(model, loader, device, weights, optim=None):
    is_train = optim is not None
    model.train(is_train)
    total, n = 0.0, 0
    preds, tgts = [], []
    for seq, line_id, y in loader:
        seq, line_id, y = seq.to(device), line_id.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            p = model(seq, line_id)
            loss = weighted_huber(p, y, weights)
            if is_train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
        total += float(loss.item()) * seq.size(0)
        n += seq.size(0)
        preds.append(p.detach().cpu().numpy())
        tgts.append(y.detach().cpu().numpy())
    return total / max(n, 1), np.concatenate(preds), np.concatenate(tgts)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cpu")
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATASET)
    parts = split(df)
    stats = compute_stats(parts["train"])
    ds_tr = VWCSequenceDataset(parts["train"], stats)
    ds_va = VWCSequenceDataset(parts["val"], stats)
    ds_te = VWCSequenceDataset(parts["test"], stats)
    print(f"sequences: train={len(ds_tr):,} val={len(ds_va):,} test={len(ds_te):,}")

    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False)

    n_lines = int(df["line_id"].max()) + 1
    cfg = LSTMConfig(input_dim=len(SEQ_FEATURE_COLS), n_lines=n_lines)
    model = LSTMForecaster(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = torch.tensor(HORIZON_LOSS_WEIGHTS, dtype=torch.float32, device=device)

    history: list[dict] = []
    best_val = float("inf")
    best_state = None
    stale = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, _, _ = run_epoch(model, dl_tr, device, weights, optim)
        va_loss, va_p, va_y = run_epoch(model, dl_va, device, weights, None)
        va_rmse = float(np.sqrt(np.mean((va_p - va_y) ** 2)))
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_rmse": va_rmse})
        print(f"ep {epoch:02d}  train={tr_loss:.4f}  val={va_loss:.4f}  val_rmse={va_rmse:.3f}")
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, va_p, va_y = run_epoch(model, dl_va, device, weights, None)
    _, te_p, te_y = run_epoch(model, dl_te, device, weights, None)
    val_metrics = metrics(va_y, va_p)
    test_metrics = metrics(te_y, te_p)
    elapsed = round(time.time() - t0, 1)

    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "stats_mean": stats.mean.tolist(),
        "stats_std": stats.std.tolist(),
        "feature_cols": SEQ_FEATURE_COLS,
    }, MODELS / "forecaster_lstm.pt")

    report = {
        "elapsed_seconds": elapsed,
        "sequences": {"train": len(ds_tr), "val": len(ds_va), "test": len(ds_te)},
        "config": cfg.__dict__,
        "batch": BATCH, "epochs": EPOCHS, "lr": LR,
        "patience": PATIENCE, "huber_delta": HUBER_DELTA,
        "horizon_weights": HORIZON_LOSS_WEIGHTS,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (REPORTS / "lstm_training_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote models/forecaster_lstm.pt  +  reports/lstm_training_report.json")
    for label in HORIZON_LABELS:
        print(f"  h={label}: val_rmse={val_metrics[f'rmse_{label}']:.3f} "
              f"test_rmse={test_metrics[f'rmse_{label}']:.3f} "
              f"test_nse={test_metrics[f'nse_{label}']:.3f}")


if __name__ == "__main__":
    main()
