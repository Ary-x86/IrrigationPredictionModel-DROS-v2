# Non-DROS v2 — Smart Irrigation AI

Research-backed rebuild of a smart-irrigation controller for a Stuard research-farm tomato plot (lat 44.1125, lon 10.411). Replaces the original Preite & Vignali (2024) MLP classifier (which had label leakage — target was computed from two of six input features) with a **soil-moisture forecaster + rule policy** pipeline. Built in two parallel tracks so we can compare empirically.

- **Track A (done):** XGBoost + LightGBM GBDT forecasters in `experiments/gbdt/`. Trains on the 37-day Stuard dataset, zero heavy deps, runs in minutes. Already ships a working forecaster and leakage audit.
- **Track B (planned):** 2-layer PyTorch LSTM rebuild in `src/` using ERA5 + Sentinel-2 + ISMN + SMAP transfer learning, MPC policy, SHAP explanations.
- **Phase 6:** head-to-head comparison of the two tracks on the held-out test week.

Full plan lives in `PROJECT_NOTES_AND_IMPROVEMENTS.md` and the approved plan file `.claude/plans/good-notice-that-the-sorted-karp.md`.

---

## What problem is this solving

The original project target `Irrigation_Decision` was computed **deterministically** from `Soil Moisture` + `Weather Forecast Rainfall` (via threshold rules in `src/03_soil_capacity_calculator.py`), which were then fed in as model inputs. The MLP was re-learning a threshold rule. Reported ~99% accuracy and Monte Carlo "savings up to 57%" were tautological.

Fix: reframe the task from **"classify irrigation action from current instant"** to **"forecast future soil VWC trajectory, then derive action from the forecast."** The target (future VWC at horizons {1, 3, 6, 12, 24}h) is no longer a deterministic function of current-instant features, so the leakage route is structurally closed.

---

## Current state

| Component | Status | Path |
|---|---|---|
| Track A GBDT features (VPD, Ra, Hargreaves ET0, GDD-Kc, lags, rolls, rain sums) | done | `experiments/gbdt/build_features.py` |
| Track A temporal splits (24h embargo) | done | `experiments/gbdt/temporal_split.py` |
| Track A XGBoost per-horizon trainer | done | `experiments/gbdt/train_xgboost.py` |
| Track A LightGBM per-horizon trainer | done | `experiments/gbdt/train_lightgbm.py` |
| Track A baselines (persistence, climatology, water-balance bucket) | done | `experiments/gbdt/baselines.py` |
| Track A stage-aware MAD rule policy | done | `experiments/gbdt/policy_rule.py` |
| Track A leakage audit (shuffle-label + permutation + ablation) | done | `experiments/gbdt/audit_leakage.py` |
| Track A evaluator + report | done | `experiments/gbdt/evaluate.py` → `reports/report.md` |
| Track B Phase 0 (legacy audit + main_controller bundle fix) | pending | — |
| Track B Phases 1–5 (ERA5/S2/ISMN/SMAP, PyTorch LSTM, MPC, SHAP) | pending | `src/` |
| Phase 6 cross-track comparison | pending | — |

Legacy Preite MLP still sits in `src/01…05_*.py` + `main_controller.py` as a reference baseline, but its label leakage is documented and it will be renamed to `models/legacy_mlp.pkl` during Phase 0.

---

## Track A results (held-out, honest numbers)

Splits: train `2023-07-28 → 2023-08-20`, val `2023-08-21 → 2023-08-27`, test `2023-08-28 → 2023-09-03`, all with 24h embargo on each side of val/test.

Forecast RMSE (soil VWC %):

| split | h | persistence | water_balance | xgboost | lightgbm |
|---|---|---:|---:|---:|---:|
| val | 1h | 2.00 | 1.99 | 1.97 | **1.70** |
| val | 3h | 3.80 | 3.75 | 3.74 | **3.50** |
| val | 6h | 4.87 | 4.75 | **3.93** | 3.88 |
| val | 12h | 5.06 | 5.03 | **3.31** | 3.34 |
| val | 24h | 5.71 | 7.31 | **3.70** | 3.68 |

GBDTs beat persistence on validation at every horizon — plan acceptance criterion met. Full table (incl. MAE/NSE + test split) in `experiments/gbdt/reports/report.md`.

Leakage audit (`reports/leakage_audit.md`):
- Shuffle-label RMSE **4.66** vs persistence **2.00** → model is not memorising labels.
- Dropping `vwc_20cm` + all VWC lags degrades val RMSE by **+0.93** → model uses non-VWC features meaningfully.
- Permutation importance top-3: `vwc_20cm` (+3.04), `hours_since_last_irrigation` (+0.54), `vwc_20cm_deriv_1h` (+0.06).

---

## Install

```bash
pip install -r requirements.txt
pip install xgboost lightgbm
```

Python 3.11+ recommended. Core deps already in `requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `requests` (for weather fetch). Track A needs `xgboost`, `lightgbm`. Track B will add `torch`, `shap`, `cdsapi`, `pystac-client`.

---

## How to run Track A

From the repo root:

```bash
# 1. Build feature table (reads data/merged_sensor_data.csv + data/open_meteo_forecast_data.csv,
#    writes experiments/gbdt/data/features.csv — 15,981 rows x 39 cols)
python -m experiments.gbdt.build_features

# 2. Sanity-check the temporal splits
python -m experiments.gbdt.temporal_split

# 3. Train both GBDT variants (one model per horizon, early-stopped on val RMSE)
python -m experiments.gbdt.train_xgboost      # ~90s total, saves xgb_h{1,3,6,12,24}h.json
python -m experiments.gbdt.train_lightgbm     # ~3min total, saves lgb_h{1,3,6,12,24}h.txt

# 4. Run leakage audit on the h=1h model (shuffle-label + permutation + ablation, ~70s)
python -m experiments.gbdt.audit_leakage

# 5. Evaluate both models + all baselines on val and test, backtest policy
python -m experiments.gbdt.evaluate
```

Outputs land in `experiments/gbdt/`:

```
experiments/gbdt/
├── data/features.csv               # feature table with targets
├── models/
│   ├── xgb_h{1h,3h,6h,12h,24h}.json
│   └── lgb_h{1h,3h,6h,12h,24h}.txt
└── reports/
    ├── xgb_training_report.json    # per-horizon train stats
    ├── lgb_training_report.json
    ├── leakage_audit.md            # + .json
    ├── evaluation.json             # full metrics + policy backtest
    └── report.md                   # human-readable Track A deliverable
```

Re-run order: steps 1–5 are dependent; re-run 1 if raw CSVs change, 3 if features change, 4–5 after any model change. Steps 4 and 5 are independent of each other.

---

## How it works

### Feature set (`build_features.py`)

- **Raw sensor** (per-line, 10-min grid): `vwc_20cm`, `soil_temp_c`, `air_temp_c`, `rh_pct`.
- **Derived physics**: `vpd_kpa` (Tetens), `ra_mj_m2_day` (FAO-56 eq. 21), `daylength_h` (FAO-56 eq. 34), `et0_hargreaves_mm_day` (Hargreaves-Samani), `etc_mm_h = kc_dynamic × et0`.
- **Phenology**: `gdd_cum` (base 10 °C from transplant 2023-05-01), `growth_stage` ∈ {initial, development, mid, late}, `kc_dynamic` interpolated across stage centers {0.40, 0.75, 1.15, 0.85}.
- **Time encoding**: `sin/cos_doy`, `sin/cos_hour`.
- **Lags + rolls**: VWC lags at {1, 3, 6, 12, 24}h (→ 6, 18, 36, 72, 144 steps), 3h rolling mean, 6h rolling std, 1h finite-difference derivative.
- **Weather**: `rain_mm_h`, rolling rain sums at 1h/6h/24h, `et0_open_meteo_mm_h`.
- **Irrigation state**: `volume_diff` (mm per 10-min), `hours_since_last_irrigation`, `line_id` embedding.
- **Targets**: `y_vwc_h{1h,3h,6h,12h,24h}` via `groupby(line).shift(-N)` — strictly future, per line.

### Splits (`temporal_split.py`)

Hard date cutoffs with 24h embargo applied to both sides of the val and test windows so rolling-window features cannot leak across the boundary.

### Models (`train_xgboost.py`, `train_lightgbm.py`)

One regressor per horizon, trained independently. XGBoost: `max_depth=6`, `lr=0.05`, up to 1000 rounds with early stopping on val RMSE. LightGBM: `num_leaves=63`, `lr=0.05`, up to 2000 rounds.

### Baselines (`baselines.py`)

- **Persistence**: `y_hat(t+h) = vwc(t)`. Strong on a stable test week.
- **Climatology**: per-line train-set mean.
- **Water-balance bucket**: `vwc + h × (rain - etc)`. Degrades at longer horizons (no irrigation knowledge).

### Policy (`policy_rule.py`)

Reads the t+3h VWC forecast; derives MAD threshold from stage (MAD fraction × TAW, TAW = FC − WP); emits `{OFF, ON_LOW=5mm, ON_HIGH=10mm}`. Hourly dedup before volume sum in `evaluate.py` to avoid counting 6 decisions per hour.

### Leakage audit (`audit_leakage.py`)

Three hard checks, all on the h=1h target:
1. **Shuffle-label** — retrain on shuffled y, must be worse than persistence. (4.66 vs 2.00: pass.)
2. **Permutation importance** — shuffle each feature on val, measure RMSE delta.
3. **Feature ablation** — drop `vwc_20cm` + all VWC-derived lags/rolls, retrain. Degradation of +0.93 RMSE shows the model uses other features too.

---

## How to test

No formal test suite yet for Track A (planned for Track B Phase 1 in `tests/`). For now:

- Re-run step 5 — `report.md` regenerates; eyeball the metrics table.
- Acceptance gate from the plan: **GBDT must beat persistence at h=1h on val**. Checked in `report.md`. Currently: persistence 2.00, xgb 1.97, lgb 1.70 → **pass**.
- Leakage gate: `leakage_audit.md` shuffle-label row must show `pass: True`.

---

## Configuration

All Track A constants live in `experiments/gbdt/config.py`:

- Geo: `LAT=44.1125`, `LON=10.411`.
- Crop: `TRANSPLANT_DATE="2023-05-01"`, `T_BASE_C=10`, `GDD_STAGE_EDGES`, `KC_BY_STAGE`.
- Splits: `TRAIN_START/END`, `VAL_*`, `TEST_*`, `EMBARGO_HOURS=24`.
- Soil: `FIELD_CAPACITY_PCT=27.41`, `WILTING_POINT_PCT=15.0`, `MAD_FRACTION_BY_STAGE`.
- Hardware: `LINE_FLOW_LPH=300`, `PUMP_POWER_KW=4.0`, `FIELD_AREA_M2=132`.
- References: `IRRIFRAME_SEASON_MM=324.5`.

---

## Research backing

Tagged per technical choice:

- **Task reframe (forecast-then-policy)** — Hamdaoui 2024 (PRISMA review of SM forecasters); Jaiswal 2025 (Smart-Drip IoT review).
- **GBDT for data-scarce tabular** — Hamdaoui 2024.
- **VPD as top-5 feature / SHAP on XGB** — Wagan 2025.
- **FAO-56 ETc, Kc curves** — Allen FAO-56; WIAI 2021 white paper.
- **Hargreaves-Samani backup ET** — Sanikhani 2018.
- **GDD-driven dynamic Kc** — Sensors 2019 (Remote Control of Greenhouse Vegetable Production).
- **Temporal holdout + embargo + leakage audits** — PRISMA Review 2025; Big-Data decision-making review.
- **2-layer LSTM (Track B)** — Hamdaoui 2024, Jaiswal 2025, Dhanke 2025 (Climate-Based AI), DLISA.
- **Transformer deferred to Phase 6** — Ikegawa 2026 (irrigation-district energy forecasting).
- **MPC policy over forecaster** — Ikegawa 2026; Novel autonomous irrigation 6G-IoT.
- **Interpretability (SHAP/LIME)** — Wagan 2025; PRISMA 2025.
- **ERA5 + transfer learning** — Hamdaoui 2024 (flags single-season generalisation gap).
- **Sentinel-2 NDVI/EVI/LAI** — Applications AI in Precision Irrigation.
- **Autoencoder anomaly (Phase 6)** — Benameur 2024.

---

## Honest limitations

- Only 37 days of Stuard data (mid-to-late-season). Model is untrustworthy at transplant or harvest phases until Track B adds ERA5 pretraining.
- Single 20 cm probe — cannot disambiguate surface wetting from deep drainage.
- Kc curves are tomato-specific, no cross-crop claim.
- "Water-savings" numbers are simulation, not field-validated.
- Policy backtest's `Stuard mm/line` column uses `volume_diff` sum, which likely reflects cumulative-counter semantics rather than per-step deltas on this feed — treat that column as a placeholder until we validate the water-meter schema in Track B.
- IRRIFRAME comparison fidelity: IRRIFRAME's internal parameters for this plot aren't public; our 324.5 mm reference is from the paper, not re-derived.

---

## Next steps

- **Track B Phase 0** — honesty check: fix `main_controller.py` bundle-unbox bug; run permutation audit on the legacy MLP; write `reports/leakage_audit_phase0.md`.
- **Track B Phase 1** — build `src/features/*`, `src/data_io/fetch_{era5,sentinel2,ismn,smap}.py`, produce `data/processed/modeling_dataset_v2.parquet`.
- **Track B Phase 2** — PyTorch LSTM forecaster + pretrain-then-finetune.
- **Track B Phase 3** — MPC policy layer over the forecast.
- **Track B Phase 4** — physics-based Monte Carlo replacing `src/05_monte_carlo_simulation.py`.
- **Track B Phase 5** — SHAP report + fixed deployment controller.
- **Phase 6** — run both tracks on the same held-out week, write `reports/track_comparison.md`, recommend the winner.
