# Track A — Gradient-Boosted Tree Forecaster (XGBoost + LightGBM)

Production-oriented forecaster for soil water content (VWC) at the Stuard site.
Tabular features, per-horizon regressors, tree-SHAP explanations.

> Shippable. Small on disk (~5 MB total), fast (~4 ms per 1000 predictions),
> and native tree-SHAP makes per-decision rationale trivial — see
> `src/deploy/inference.py`.

---

## 1. What the model achieves

**Task:** regress future soil VWC at `h ∈ {1h, 3h, 6h, 12h, 24h}` from a flat
feature vector built strictly from past-only measurements (no target leakage).

**Observed result** (Stuard val, RMSE %VWC):

| horizon | persistence | xgboost | lightgbm |
| ---: | ---: | ---: | ---: |
| 1h  | 2.00 | 1.97 | **1.70** |
| 3h  | 3.80 | 3.74 | **3.50** |
| 6h  | 4.87 | 3.93 | **3.88** |
| 12h | 5.06 | 3.31 | 3.34 |
| 24h | 5.71 | **3.68** | 3.68 |

LightGBM beats the persistence baseline at every horizon on val. The held-out
test week (2023-08-28 → 2023-09-03) is unusually stable so persistence hits
NSE 0.94 at h=1h; see `reports/track_comparison.md` for the honest reading.

**Policy output:** the h=3h regressor feeds a stage-aware Management Allowed
Depletion (MAD) rule in `src/policy/policy_rule.py` that emits
`{OFF, ON_LOW=5 mm, ON_HIGH=10 mm}` per hour. A cost-minimizing MPC
(`src/policy/policy_mpc.py`) uses all three h∈{1,3,6}h forecasts.

---

## 2. Why GBDT here (research backing)

| Choice | Paper | What it justifies |
| --- | --- | --- |
| XGBoost + LightGBM on 37-day Stuard slice | **Hamdaoui 2024** (PRISMA review) | Tree ensembles outperform deep models on *data-scarce* tabular irrigation tasks. |
| Tree-SHAP for farmer-facing rationale | **Wagan 2025** (SHAP + LIME over XGB) | Same combination Wagan uses to build interpretable irrigation decisions. |
| VPD in the top-5 features | **Wagan 2025** | VPD consistently ranks high in SHAP for irrigation targets. |
| Dynamic Kc × ET0 via GDD | Allen FAO-56 + Water-Use-Efficiency white paper + *Remote Control of Greenhouse Vegetable Production* (Sensors 2019) | Stage-dependent crop coefficient is the correct canonical transformation. |
| Hargreaves-Samani backup ET | **Sanikhani 2018** | Temperature-only ET0 when full-set radiation is missing. |

Full list in `README.MD` § Research backing.

---

## 3. Code map (where each piece lives)

```
experiments/gbdt/
  config.py            — LAT/LON, crop/soil constants, split dates, horizons
  build_features.py    — raw CSV → features.csv (39 cols)
  temporal_split.py    — train/val/test + 24h embargo
  baselines.py         — persistence, climatology, water-balance
  train_xgboost.py     — one booster per horizon
  train_lightgbm.py    — one booster per horizon
  policy_rule.py       — MAD rule on forecasted VWC
  audit_leakage.py     — shuffle-label, permutation, ablation
  evaluate.py          — writes reports/report.md + evaluation.json
  models/              — xgb_h{1,3,6,12,24}h.json, lgb_h{...}.txt
  reports/             — report.md, *_training_report.json, leakage_audit.md
  data/features.csv    — flat feature table
```

All Track A code is isolated from `src/`. It has no torch dependency and can
be trained from scratch in ~90 s total on a laptop CPU.

---

## 4. How the features are built

**Source:** `data/merged_sensor_data.csv` (10-min cadence per irrigation line)
joined with the hourly Open-Meteo archive.

**Formulas** (all in `experiments/gbdt/build_features.py`, mirrored in
`src/features/*`):

### 4.1 Vapour Pressure Deficit (Tetens form)

```
es = 0.6108 · exp(17.27·T / (T + 237.3))     # saturation vapour pressure [kPa]
ea = es · RH/100                             # actual vapour pressure   [kPa]
vpd = es − ea                                # deficit                   [kPa]
```

At T=25 °C, RH=50 % this evaluates to ~1.584 kPa (tested in
`tests/test_features.py`). VPD drives transpiration demand — Wagan 2025 finds
it in the top-5 SHAP features.

### 4.2 Extraterrestrial radiation Ra (FAO-56 eq. 21)

```
dr = 1 + 0.033·cos(2π·doy/365)
δ  = 0.409·sin(2π·doy/365 − 1.39)
φ  = lat · π/180
ωs = arccos(−tan φ · tan δ)
Ra = (24·60/π) · Gsc · dr · (ωs·sin φ·sin δ + cos φ·cos δ·sin ωs)     # MJ/m²/day
Gsc = 0.0820 MJ/m²/min
```

### 4.3 Daylength (FAO-56 eq. 34)

```
N = (24/π) · ωs        # hours
```

### 4.4 Hargreaves-Samani backup ET0 (Sanikhani 2018)

```
ET0 = 0.0023 · (Tmean + 17.8) · √(Tmax − Tmin) · (Ra / 2.45)         # mm/day
```

Used when Open-Meteo's full-set ET is unavailable.

### 4.5 GDD + dynamic Kc

```
GDD_day = max(Tmean − 10 °C, 0)
GDD_cum = Σ GDD_day from transplant (2023-05-01)

stage    = piecewise  { initial:   GDD < 350
                        developm.: 350 ≤ GDD < 700
                        mid:       700 ≤ GDD < 1100
                        late:      GDD ≥ 1100 }

Kc_at_stage_edges = {initial: 0.40, development: 0.75, mid: 1.15, late: 0.85}
Kc_dynamic        = piecewise-linear interpolation across stages

ETc = Kc_dynamic × ET0
```

### 4.6 VWC lags, rolls, derivatives

```
vwc_20cm_lag_{6,18,36,72,144}s   # 10-min steps → 1h, 3h, 6h, 12h, 24h
vwc_20cm_roll_mean_3h            # 18-step rolling mean
vwc_20cm_roll_std_6h             # 36-step rolling std
vwc_20cm_deriv_1h                # finite-difference drying rate
```

### 4.7 Rain rolls and irrigation clock

```
rain_mm_{1h,6h,24h}_sum          # rolling sums
hours_since_last_irrigation      # clock reset by nonzero irrigation_volume
```

### 4.8 Time encodings + identity

```
sin_doy,  cos_doy                # day-of-year periodic
sin_hour, cos_hour               # hour-of-day periodic
line_id                          # categorical irrigation line
```

Full feature vector (30 cols): see `FEATURE_COLS` in
`experiments/gbdt/train_xgboost.py:20-33`.

---

## 5. Temporal splits with 24h embargo

`experiments/gbdt/temporal_split.py:22-38`.

```
train: 2023-07-28 00:00 .. 2023-08-20 00:00  (embargo shrinks by 24h on upper edge)
val:   2023-08-21 24h .. 2023-08-27 24h      (embargo on both sides)
test:  2023-08-28 24h .. 2023-09-03 23:59    (untouched)
```

Embargo kills any adjacency leakage between fold boundaries — a VWC value
within 24h of the boundary could be strongly correlated with samples across
the cut, which would be a subtler form of the leakage we killed by switching
the target.

---

## 6. Model specification

Both trees are one-regressor-per-horizon.

### 6.1 XGBoost (`train_xgboost.py:35-48`)

```python
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
)
NUM_BOOST_ROUND = 1000
EARLY_STOPPING  = 50   # rounds of no val-RMSE improvement
```

Per-horizon best-iteration (from `xgb_training_report.json`):

| horizon | best_iter | train s |
| --- | ---: | ---: |
| 1h  |  53 | 16.6 |
| 3h  |  26 | 14.6 |
| 6h  | 128 | 26.7 |
| 12h | 103 | 23.3 |
| 24h |   7 |  7.9 |

Total wall-clock: ~90 s for five boosters.

### 6.2 LightGBM (`train_lightgbm.py`)

```
num_leaves = 63     # ≈ depth 6
learning_rate = 0.05
feature_fraction = 0.9
bagging_fraction = 0.9
min_child_samples = 20
reg_lambda = 1.0
num_boost_round = 2000
early_stopping_rounds = 50
```

LightGBM's leaf-wise growth explains why it edges XGBoost at h∈{1h,3h,6h} —
it can capture fine-grained VWC-gradient splits that depth-limited XGBoost
cannot.

### 6.3 Loss — squared error

We use `reg:squarederror` (not Huber) for the GBDT path. Reasoning:
- VWC residuals on Stuard are near-Gaussian (no heavy outliers on this feed).
- Squared error is closed-form for GBDT second-order boosting; Huber needs a
  custom objective and trades convergence speed for a mild robustness gain
  we don't need. Huber *is* used for the LSTM head where per-horizon
  magnitudes are on very different scales.

---

## 7. Leakage audits

`experiments/gbdt/audit_leakage.py` runs three checks; all pass on current
artifacts (`experiments/gbdt/reports/leakage_audit.md`).

1. **Shuffle-label sanity:** randomly permute targets, train a fresh booster,
   require that val RMSE is worse than the persistence baseline (otherwise
   the model is exploiting something other than signal). On our feed
   shuffle-label RMSE is ~4.66 vs persistence 2.00 — **pass**.

2. **Permutation importance:** shuffle one feature at a time across the val
   set and measure val-RMSE delta. No single feature explains the forecast
   alone; top drivers are `vwc_20cm`, `hours_since_last_irrigation`,
   `etc_mm_h` — physically sensible for soil drying.

3. **Ablation:** drop `vwc_20cm` + all VWC lags, retrain. RMSE degrades by
   ~+0.93 at h=1h — the model is not just reading the current VWC off the
   feature vector.

---

## 8. Baselines inside Track A

`experiments/gbdt/baselines.py` ships three sanity baselines the trees must
beat:

- **Persistence:** `vwc(t + h) = vwc(t)`. This is the hard floor.
- **Climatology:** seasonal mean at (doy, hour). Tests whether the model
  learns anything beyond "mid-August afternoons look like this."
- **Water-balance bucket:** `dM = rain − ETc + irrigation`, clipped to
  `[WP, FC]`. Physical first-principles forecast; proves whether the tree is
  learning more than the naive FAO-56 bucket.

See `experiments/gbdt/reports/report.md` for all baselines × all horizons.

---

## 9. SHAP rationale (Wagan 2025 pattern)

`src/explain/shap_report.py` loads `xgb_h3h.json`, runs `shap.TreeExplainer`
over a sample of test rows, and emits per-decision top-3 drivers.

Mean |SHAP| on 20 sampled test decisions (from `reports/shap_report.md`):

| feature | mean \|SHAP\| |
| --- | ---: |
| vwc_20cm | 1.40 |
| hours_since_last_irrigation | 0.96 |
| etc_mm_h | 0.33 |
| line_id | 0.27 |
| air_temp_c | 0.22 |

This agrees with Wagan's finding that VWC + evaporative demand proxies
dominate irrigation rationale.

---

## 10. Deployment wrapper

`src/deploy/inference.py` turns a feature row into a farmer-facing JSON:

```json
{
  "action": "OFF",
  "volume_mm": 0.0,
  "predicted_vwc_3h": 24.26,
  "mad_threshold": 22.45,
  "growth_stage": "development",
  "reason_top3_features": [
    {"feature": "hours_since_last_irrigation", "value": 228.3, "shap": 1.05, "direction": "wetter"},
    {"feature": "air_temp_c",                  "value": 18.4,  "shap": -0.41, "direction": "drier"},
    {"feature": "line_id",                     "value": 2.0,   "shap": -0.29, "direction": "drier"}
  ],
  "confidence": 0.146
}
```

`confidence = min(|pred − MAD| / TAW, 1)` — distance from the decision
boundary normalized by Total Available Water. 0 at the threshold, 1 at FC or WP.

---

## 11. How to reproduce

```
cd experiments/gbdt
python -m experiments.gbdt.build_features
python -m experiments.gbdt.train_xgboost
python -m experiments.gbdt.train_lightgbm
python -m experiments.gbdt.audit_leakage
python -m experiments.gbdt.evaluate    # writes reports/report.md
```

Then the cross-track comparison that includes both XGBoost, LightGBM, and the
Track B LSTM:

```
python -m src.eval.compare_tracks      # writes reports/track_comparison.md
```

---

## 12. Honest limits

- Single season of Stuard data. Cross-season generalization untested.
- `line_id` sits in the top-5 SHAP drivers, which means the model has learned
  per-line biases. That's fine for this site but won't transfer without
  re-training.
- Forecast RMSE at h=1h test (~1.0-1.4) is *worse* than persistence (0.83).
  On a stable slice, the best forecaster is "do nothing." ERA5/ISMN
  pretraining (credential-gated, deferred) is the lever that would close
  this gap on non-stable slices.
- MAD thresholds are tomato-specific (`src/config.py:46`).
