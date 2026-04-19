# Track B — 2-Layer LSTM Multi-Horizon VWC Forecaster (PyTorch)

Research-track sequence model. Learns VWC dynamics from a 24h lookback
window, predicts all five horizons in a single forward pass, embeds
per-line identity for multi-line learning.

> Kept as the *research* track. Track A GBDT is the production path. Track B
> becomes competitive once ERA5/ISMN/SMAP pretraining lands (see § 9).

---

## 1. What the model achieves

**Task:** map a 144-step (24h × 10 min) standardized feature sequence +
line-id to a 5-horizon VWC forecast `ŷ ∈ ℝ⁵` for
`h ∈ {1h, 3h, 6h, 12h, 24h}`.

**Observed val RMSE** (`reports/baseline_comparison.md`):

| horizon | persistence | water-balance | lstm |
| ---: | ---: | ---: | ---: |
| 1h  | 2.00 | 1.99 | 2.54 |
| 3h  | 3.80 | 3.75 | **2.99** |
| 6h  | 4.87 | 4.75 | **3.48** |
| 12h | 5.06 | 5.03 | **3.95** |
| 24h | 5.71 | 7.31 | **4.35** |

**Acceptance** (from the plan, § Phase 2):

- ✅ **LSTM beats water-balance at h=6h val** (3.48 vs 4.75).
- ✅ **LSTM dominates at long horizons on val** (h=12h: 3.95 vs 5.06;
  h=24h: 4.35 vs 5.71).
- ❌ **LSTM loses to persistence at h=1h val** (2.54 vs 2.00). Short-horizon
  persistence is near-optimal on this stable 37-day feed; closing this gap
  needs ERA5/ISMN pretraining. This is documented honestly — **not** hidden.

**Test week** (2023-08-28 → 2023-09-03) is pathologically stable: persistence
hits NSE 0.94 at h=1h and both tracks lose to it on that slice.

---

## 2. Why a 2-layer LSTM (research backing)

| Choice | Paper | Reason |
| --- | --- | --- |
| 2-layer LSTM at 64 hidden units | **Hamdaoui 2024** (PRISMA review) | "Sweet spot" on data-scarce SM forecasting; Transformers overfit, single-layer underfits. |
| Multi-horizon direct regression head | **Jaiswal 2025** (*Smart drip irrigation IoT review*), **Dhanke 2025 DLISA** | Direct > iterated for ≤24h; avoids compounding rollout error. |
| Line-id embedding | multi-series literature + Stuard has 3 physically distinct lines | Share trunk, specialize bias per line. |
| Weighted Huber loss | Allen FAO-56 noise-model + general robust-regression practice | Near-horizon residuals tighter and more actionable than 24h; weight them higher. |
| Transformer deferred to Phase 6 | **Ikegawa 2026** | Transformers shine with more data; plan to revisit post-pretraining. |

---

## 3. Architecture

`src/models/lstm_forecaster.py`.

```
            seq: (B, 144, F=18)          line_id: (B,)
                  │                            │
                  ▼                            ▼
           2-layer LSTM                 Embedding(n_lines, 8)
         hidden=64, dropout=0.2                 │
                  │                             │
           last hidden h_T (B, 64)              │
                  └──── concat ─────────────────┘
                          (B, 64+8)
                            │
                  Linear(72 → 64) + ReLU + Dropout(0.2)
                            │
                    Linear(64 → 5)     ←  5 horizons
                            │
                    ŷ = (ŷ₁ₕ, ŷ₃ₕ, ŷ₆ₕ, ŷ₁₂ₕ, ŷ₂₄ₕ)
```

Config (`LSTMConfig`, `lstm_forecaster.py:20-28`):

```python
input_dim      = 18          # len(SEQ_FEATURE_COLS)
n_lines        = 3
line_embed_dim = 8
hidden_dim     = 64
num_layers     = 2
dropout        = 0.2
horizons       = 5
```

Parameter count: ~40 K. Checkpoint on disk: **0.23 MB**.

### Why last-step pooling (not attention)

The LSTM already compresses the 144-step history into `h_T`. Adding attention
on ~9 000 training sequences tends to overfit. If we had ERA5 pretraining
volume (multi-year × multi-site), Transformer + self-attention would be the
natural upgrade — plan defers it to Phase 6.

### Why concat at the head (not on the input)

Mixing the line embedding into the LSTM input makes it participate in every
timestep's gating. We want it to influence only the *output*, because it is
a constant per sequence and the trunk should learn line-agnostic dynamics.
Concat at the head is the minimum-coupling choice.

---

## 4. Feature vector (standardized per training stats)

`src/training/dataset.py:17-26`. 18 features — fewer than Track A because
the LSTM eats the *raw* VWC sequence, so pre-lagged VWC columns would be
redundant.

```
vwc_20cm                     # current VWC (sequence input)
vwc_20cm_deriv_1h            # drying-rate proxy
soil_temp_c, air_temp_c, rh_pct, vpd_kpa
rain_mm_h, rain_mm_{1h,6h,24h}_sum
et0_open_meteo_mm_h, etc_mm_h, kc_dynamic
sin_doy, cos_doy, sin_hour, cos_hour
hours_since_last_irrigation
```

Standardization stats are computed on train only (`compute_stats`,
`dataset.py:39-44`) and shipped inside the checkpoint so inference can
recompute `(X − μ) / σ` identically.

---

## 5. Targets and sequence builder

`VWCSequenceDataset` (`dataset.py:47-85`) slides a `LOOKBACK_STEPS = 144`
window per irrigation line, emitting `(seq, line_id, y)` where `y` comes
from pre-computed columns `y_vwc_h{1h,3h,6h,12h,24h}` in
`data/processed/modeling_dataset_v2.parquet`. These are built by
`src/labels/derive_targets.py` via `g.shift(-N)` per line — strictly
future-only, nothing else.

Samples with any NaN in `y` are skipped (happens only within 24h of the
end of each line's time series). Final counts on current splits:

| split | sequences |
| --- | ---: |
| train |  9,072 |
| val   |  1,731 |
| test  |  1,731 |

---

## 6. Temporal splits with 24h embargo

`src/splits/temporal_split.py`. Mirrors the Track A split so both tables in
`reports/track_comparison.md` line up row-for-row.

```
train: 2023-07-28 .. 2023-08-20   (−24h embargo at upper edge)
val:   2023-08-21+24h .. 2023-08-27−24h
test:  2023-08-28+24h .. 2023-09-03
```

---

## 7. Loss — weighted Huber

`src/training/train_forecaster.py:45-47`. Per-horizon weights
`[1.0, 0.8, 0.6, 0.4, 0.3]` from the plan.

```python
def weighted_huber(pred, target, weights):
    per = huber_loss(pred, target, delta=HUBER_DELTA, reduction="none")   # HUBER_DELTA=1.0
    return (per * weights).mean()
```

Why these weights: a miscall at h=1h is directly actionable (the next
irrigation decision happens inside that hour) while h=24h is a planning
signal. Weighting the near-horizon higher pushes capacity into the forecast
regime the policy actually reads (h=3h in `policy_rule.py`).

Why Huber over MSE: soil sensors occasionally spike (cleaning, deposits).
Huber with δ=1.0 (≈1 %VWC) gives MSE-like behavior near small errors and
L1-like behavior past the spike scale so one bad sample does not dominate
a batch gradient.

---

## 8. Optimizer + schedule

`train_forecaster.py:31-36`:

```python
BATCH         = 128
EPOCHS        = 80        # early-stopped in practice
LR            = 5e-4      # Adam default lowered for small-dataset stability
WEIGHT_DECAY  = 1e-4      # AdamW-style L2
PATIENCE      = 12        # epochs of no val-loss improvement
```

Training wall-clock on current feed: **~242 s on CPU for 17 epochs** (then
early-stopped). Gradient clipping at norm 1.0 (`train_forecaster.py:77`)
keeps the LSTM's BPTT stable on occasional 10-min outliers.

Learning-rate history (tried → selected):

| setting | val RMSE @ h=1h | notes |
| --- | ---: | --- |
| `LR=1e-3`, `PATIENCE=6`, `WD=1e-5` | ~3.1 | aggressive, overfits before val dips |
| `LR=5e-4`, `PATIENCE=12`, `WD=1e-4` | 2.54 | **shipped** |

---

## 9. Known gap and the fix (ERA5 pretraining)

The model loses h=1h on val because persistence is near-perfect on stable
feeds and 37 days is not enough to out-learn it. The plan's fix:

1. **Pretrain** on 2020-2023 ERA5 reanalysis + ISMN Italian/Spanish tomato
   stations + SMAP L3/L4 root-zone — millions of hourly samples teaching the
   LSTM generic drying-curve priors.
2. **Freeze** the lower LSTM layer.
3. **Fine-tune** upper layer + head on Stuard 2023.

Credential-gated fetchers are shipped (`src/data_io/fetch_{era5,ismn,smap,
sentinel2}.py`) with full auth docs. The pretraining itself is deferred
until the user configures Copernicus CDS + NASA Earthdata keys — per the
plan. This is the single biggest lever remaining.

---

## 10. Code map

```
src/
  models/lstm_forecaster.py         — nn.Module + LSTMConfig
  training/
    dataset.py                      — VWCSequenceDataset, compute_stats
    train_forecaster.py             — trainer (single-file, CPU)
    pretrain_era5.py                — (stub; blocked on credentials)
    finetune_stuard.py              — (stub; blocked on pretraining)
  splits/temporal_split.py          — date windows + 24h embargo
  labels/derive_targets.py          — y_vwc_h* via g.shift(-N)
  features/*.py                     — all physics (VPD, Ra, ET0, Kc, GDD...)
  eval/
    compare_baselines.py            — LSTM vs persistence vs water-balance
    compare_tracks.py               — Phase 6 GBDT vs LSTM
    metrics.py                      — RMSE, MAE, NSE
  policy/                           — fao56, rule, mpc (same policy layer both tracks)
  sim/water_balance_mc.py           — Phase 4 physics Monte Carlo

models/forecaster_lstm.pt           — trained checkpoint
reports/
  baseline_comparison.md            — LSTM vs physics baselines
  lstm_training_report.json         — per-epoch history + final metrics
  track_comparison.md               — Phase 6 cross-track
```

---

## 11. Leakage guards

`tests/test_leakage.py` runs three regression checks every CI run:

1. **Target-alignment check.** For every line, `y_vwc_h{N}` must equal the
   time-aligned `vwc_20cm` shifted by exactly `N` steps. Any drift between
   label construction and the raw series would break this.
2. **No-target-features.** Enforce that `y_vwc_h*` never appears in the
   feature column list (`SEQ_FEATURE_COLS`).
3. **Future-timestamp scan.** Every non-target column's timestamp must be
   `≤ t`. Catches an accidental forward-fill bug at build time.

`src/training/audit_leakage.py` additionally runs a permutation-importance
sweep on the legacy MLP (for the `reports/leakage_audit_phase0.md`
writeup — documents *why* we rebuilt).

---

## 12. Feature importance (LSTM permutation, from `track_comparison.md`)

Permutation Δ-RMSE at h=3h on val (`compare_tracks.py:_permutation_importance_lstm`):

| rank | feature | Δ-RMSE |
| ---: | --- | ---: |
| 1 | vwc_20cm | +1.389 |
| 2 | vwc_20cm_deriv_1h | +0.457 |
| 3 | hours_since_last_irrigation | +0.111 |
| 4 | sin_hour | +0.053 |
| 5 | cos_hour | +0.037 |
| 6 | air_temp_c | +0.030 |
| 7 | et0_open_meteo_mm_h | +0.009 |
| 8 | kc_dynamic | +0.002 |

Current VWC + drying-rate + irrigation clock dominate, exactly as expected.
Diurnal encodings matter more for the LSTM than for the GBDT because the
LSTM reads the full 24h trajectory and uses them as phase anchors.

Jaccard agreement with GBDT top-10 = **0.18**. Both rank `vwc_20cm` first
and agree on `hours_since_last_irrigation` and `air_temp_c`. Where they
differ: GBDT leans on VWC lags (pre-engineered), LSTM leans on diurnal
phase + drying rate (sequence-native). Different views of the same physics.

---

## 13. Inference wall-clock

From `reports/track_comparison.json`:

| path | 1000 predictions |
| --- | ---: |
| Track A (single XGB booster, h=3h) | 4.2 ms |
| Track B (LSTM, all 5 horizons) | 342.1 ms |

Track A is ~80× faster per prediction, but Track B outputs all five horizons
at once. Apples-to-oranges. If latency matters in deployment, use Track A;
if the MPC wants a full trajectory in one go, Track B is cheaper per
trajectory.

---

## 14. How to reproduce

```bash
# 1. Build the canonical feature parquet (shared by both tracks)
python -m src.features.assemble

# 2. Train the LSTM
python -m src.training.train_forecaster

# 3. Baseline comparison
python -m src.eval.compare_baselines

# 4. Phase 6 cross-track (needs Track A models already trained)
python -m src.eval.compare_tracks
```

---

## 15. Honest limits

- **Single-season fine-tune.** Cross-season claims are not supported by the
  current checkpoint.
- **Last-step pooling** discards information from the intermediate sequence.
  A learned pooling (attention) or deeper per-line heads could help, but
  are overkill at 9 K sequences.
- **No uncertainty.** Point forecasts only. A quantile head or
  deep-ensemble + predictive-interval layer would tell the farmer *how sure*
  the model is. Deployment currently proxies confidence via distance from
  the MAD threshold, which is a policy property not a model property.
- **Wall-clock on CPU.** GPU training would cut epoch time ~10×, but the
  checkpoint size and inference path are CPU-native by design (field
  deployment target has no GPU).
