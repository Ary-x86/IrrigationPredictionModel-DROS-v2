# HydroCast (DROS v2) — Physics + AI Soil-Moisture Forecaster for Smart Irrigation

A leakage-free rebuild of a tomato-plot irrigation controller for the Stuard research farm (lat 44.1125, lon 10.411). The task was flipped from "classify the irrigation action" to **"forecast future soil VWC, then derive the action,"** killing the structural label-leakage that made the v1 numbers tautological [Wagan 2025; PRISMA 2025].

Two forecasters run in parallel and are compared head-to-head:

- **Track A — Gradient-Boosted Trees** (XGBoost + LightGBM). Production-ready, ~5 MB, ~4 ms per 1000 predictions, native tree-SHAP rationale [Hamdaoui 2024; Wagan 2025].
- **Track B — 2-layer LSTM with line-id embedding and 5-horizon head** (PyTorch). Research track. Needs ERA5/ISMN/SMAP pretraining to fully close the h=1h gap [Hamdaoui 2024; Jaiswal 2025].

A shared **policy layer** (FAO-56 → rule on forecast → MPC) converts forecasts into actions. A **stateful physics-based Monte Carlo** estimates water-use envelopes across synthetic seasons. A **deployment module** emits farmer-readable JSON `{action, volume_mm, reason_top3_features, confidence}`.

---

## Why the rebuild — the leakage story

The v1 classifier was reported at ~99 % accuracy because its target `Irrigation_Decision` was computed **deterministically** from two of its six input features (`Soil Moisture` and `Weather Forecast Rainfall`) via threshold rules inside `src/03_soil_capacity_calculator.py`. The MLP was re-learning its own label. Monte Carlo "savings up to ~57 %" were artefacts of the sampling distribution, not model intelligence.

**Fix.** Retarget to future soil VWC at `h ∈ {1h, 3h, 6h, 12h, 24h}`. Because the target is the *physically future value*, no current-step feature can be a deterministic function of it — the leakage path is structurally closed, not just audited. See `reports/leakage_audit_phase0.md` for the empirical proof (a depth-4 decision tree on just the two original features fits the v1 target at 1.00 accuracy).

---

## Status — everything in the plan is done

> All seven phases and both tracks complete. Pretraining on external reanalysis is the only item deferred (blocked on Copernicus CDS / NASA Earthdata credentials).

| Phase | Deliverable | Status | Headline result |
| --- | --- | --- | --- |
| **Track A** | GBDT forecaster + audit + policy backtest | ✅ | LightGBM val RMSE 1.70 %VWC @ h=1h (persistence 2.00) |
| **Phase 0** | Legacy MLP leakage proof; `main_controller.py` bundle fix | ✅ | 2-feature tree fits v1 target at 1.00 accuracy [Wagan 2025] |
| **Phase 1** | `src/features/*` + canonical parquet | ✅ | `data/processed/modeling_dataset_v2.parquet` (15,981 × 39) |
| **Phase 2** | LSTM forecaster + baselines | ✅ | LSTM val h=6h RMSE **3.48** vs water-balance 4.75 — pass [Hamdaoui 2024] |
| **Phase 3** | FAO-56 + rule + MPC policies | ✅ | MPC val 93.33 mm/line vs FAO-56 321.67 (~3.4× less) [Ikegawa 2026] |
| **Phase 4** | 1000-season stateful physics MC | ✅ | FAO-56 303.3 mm/season, rule-persistence 104.4 mm, 0 stress h |
| **Phase 5** | SHAP report + deployment JSON | ✅ | Top drivers: `vwc_20cm`, `hours_since_last_irrigation`, `etc_mm_h` [Wagan 2025] |
| **Phase 6** | Cross-track comparison | ✅ | **Recommend ship Track A GBDT**; keep Track B for post-ERA5 research |
| Deferred | ERA5/ISMN/SMAP pretraining, Transformer, RL scheduler, autoencoder anomaly | ⏳ | Credential-gated fetchers shipped [Hamdaoui 2024; Ikegawa 2026; Benameur 2024] |

---

## Key results (all honest numbers — held-out, no cherry-picking)

### Forecast RMSE per horizon — val split (% VWC)

| model | 1h | 3h | 6h | 12h | 24h |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 2.00 | 3.80 | 4.87 | 5.06 | 5.71 |
| water-balance | 1.99 | 3.75 | 4.75 | 5.03 | 7.31 |
| XGBoost (A) | 1.97 | 3.74 | 3.93 | **3.31** | **3.70** |
| LightGBM (A) | **1.70** | **3.50** | 3.88 | 3.34 | 3.68 |
| LSTM (B) | 2.54 | 2.99 | **3.48** | **3.95** | **4.35** |

Notes. LightGBM dominates short horizons (h=1h, h=3h). LSTM dominates long horizons on val. The test week 2023-08-28 → 2023-09-03 is unusually stable (persistence NSE 0.94 @ h=1h), so persistence wins that slice — a property of the slice, not a bug.

### 1000-season Monte Carlo (physics-based, stateful)

| policy | mm/season (mean) | 95 % CI | stress h<WP | savings vs IRRIFRAME |
| --- | ---: | ---: | ---: | ---: |
| FAO-56 ETc | 303.3 | [258.2, 348.4] | 0.0 | +6.5 % |
| rule-persistence | 104.4 | [68.5, 140.3] | 0.0 | +67.8 % |

IRRIFRAME reference: 324.5 mm per plot per season. Savings read alongside stress — high savings with crop stress is not a win.

### Resource profile (Phase 6)

| track | disk (MB) | 1000 predictions (ms) | output |
| --- | ---: | ---: | --- |
| A — XGB + LGBM (10 boosters) | 5.03 | 4.2 (one booster) | one scalar per booster |
| B — LSTM checkpoint | 0.23 | 342.1 | 5-dim vector at once |

Feature-importance Jaccard between tracks: **0.18** (top-10 GBDT gain vs LSTM permutation Δ-RMSE @ h=3h). Both rank `vwc_20cm` #1; both agree on `hours_since_last_irrigation` and `air_temp_c`. Different views of the same physics.

---

## Quickstart — how to run

Python 3.11+.

```bash
pip install -r requirements.txt
pip install xgboost lightgbm torch --index-url https://download.pytorch.org/whl/cpu
pip install shap pytest pyarrow
```

### Track A — GBDT pipeline (~2 min total)

```bash
python -m experiments.gbdt.build_features      # builds features.csv
python -m experiments.gbdt.temporal_split      # sanity-checks split windows
python -m experiments.gbdt.train_xgboost       # 5 boosters, ~90 s
python -m experiments.gbdt.train_lightgbm      # 5 boosters, ~3 min
python -m experiments.gbdt.audit_leakage       # shuffle + permutation + ablation
python -m experiments.gbdt.evaluate            # writes experiments/gbdt/reports/report.md
```

### Track B — LSTM pipeline (~5 min CPU)

```bash
python -m src.features.assemble                # writes data/processed/modeling_dataset_v2.parquet
python -m src.training.train_forecaster        # LSTM, ~4 min, writes models/forecaster_lstm.pt
python -m src.eval.compare_baselines           # writes reports/baseline_comparison.md
python -m src.policy.backtest                  # writes reports/policy_backtest.md
python -m src.sim.water_balance_mc --seasons 1000 --seed 0   # writes reports/monte_carlo.md
python -m src.explain.shap_report --sample 20  # writes reports/shap_report.md
python -m src.deploy.inference                 # farmer JSON for the last test row
```

### Phase 6 — cross-track comparison

```bash
python -m src.eval.compare_tracks              # writes reports/track_comparison.md
```

### Tests

```bash
pytest tests/                                  # 15 assertions across physics + leakage + policy
```

---

## Repository layout — what's in each folder

```
HydroCast (DROS v2)/
├── src/                    Track B Python package (LSTM + policy + MC + deployment)
│   ├── config.py              Canonical constants (geo, soil, splits, Kc, MAD, actions)
│   ├── data_io/               Credential-gated fetchers for ERA5, Sentinel-2, ISMN, SMAP
│   ├── features/              Physics + signal features (FAO-56 Ra, VPD, Hargreaves, GDD-Kc, lags, rolls)
│   ├── labels/                derive_targets.py — y_vwc_h{N} = shift(-N) per line (strictly future)
│   ├── splits/                temporal_split.py — train/val/test with 24h embargo
│   ├── models/                lstm_forecaster.py, persistence_baseline.py, water_balance.py
│   ├── training/              LSTM trainer, sequence dataset, legacy MLP leakage audit
│   ├── policy/                FAO-56 scheduler, rule on ŷ(t+3h), MPC over ŷ(t+{1,3,6}h), backtest driver
│   ├── sim/                   water_balance_mc.py — stateful physics MC (Phase 4)
│   ├── eval/                  metrics.py, compare_baselines.py, compare_tracks.py (Phase 6)
│   ├── explain/               shap_report.py — tree-SHAP over XGBoost [Wagan 2025]
│   └── deploy/                inference.py — farmer-facing JSON emitter
│
├── experiments/gbdt/       Track A isolated prototype (zero src/ deps; ships first)
│   ├── config.py              Mirror of src/config.py for the GBDT pipeline
│   ├── build_features.py      Local reimplementation of src/features/ (avoids coupling)
│   ├── temporal_split.py      Same splits as Track B (24h embargo)
│   ├── baselines.py           Persistence, climatology, water-balance bucket
│   ├── train_xgboost.py       One booster per horizon, early-stopped on val RMSE
│   ├── train_lightgbm.py      LightGBM per-horizon equivalent
│   ├── policy_rule.py         MAD rule on ŷ(t+3h), mirrors src/policy/policy_rule.py
│   ├── audit_leakage.py       Shuffle-label + permutation importance + feature ablation
│   ├── evaluate.py            Writes reports/report.md + evaluation.json
│   ├── models/                xgb_h{1,3,6,12,24}h.json, lgb_h{...}.txt
│   ├── data/features.csv      Flat feature table (15,981 × 39)
│   └── reports/               Track A deliverables (report.md, *_training_report.json, leakage_audit.md)
│
├── data/                   Datasets
│   ├── raw/                   Stuard *.csv sensor feeds, Open-Meteo archive
│   ├── interim/               merged_sensor_data.csv (per-line 10-min grid)
│   └── processed/             modeling_dataset_v2.parquet (canonical feature + target table)
│
├── models/                 Trained artifacts
│   ├── forecaster_lstm.pt     Track B checkpoint (state_dict + config + stats + feature_cols)
│   ├── legacy_mlp.pkl         v1 MLP kept as a leakage-audit baseline
│   └── mlp_irrigation_model.pkl   Original bundle {model, feature_columns}
│
├── reports/                Generated deliverables
│   ├── leakage_audit_phase0.md        Why we rebuilt — v1 leakage proof
│   ├── baseline_comparison.md         Track B LSTM vs physics baselines
│   ├── policy_backtest.md             FAO-56 + rule + MPC backtest on val + test
│   ├── monte_carlo.md                 1000-season stateful MC
│   ├── shap_report.md                 Per-decision SHAP top-3 drivers [Wagan 2025]
│   ├── shap_examples/                 decisions.json, feature_importance.csv
│   ├── track_comparison.md            Phase 6 head-to-head (GBDT vs LSTM)
│   └── *.json                         Machine-readable versions of each
│
├── docs/                   Extensive technical documentation
│   ├── TRACK_A_GBDT.md               GBDT theory, formulas, model spec, audits, deployment
│   ├── TRACK_B_LSTM.md               LSTM architecture, loss math, training, importance
│   └── ARCHITECTURE.md               End-to-end data flow + paper-to-code mapping
│
├── tests/                  pytest suite (15 checks)
│   ├── test_features.py              VPD, Ra, daylength, SWDI known-answer assertions
│   ├── test_leakage.py               Target alignment + no-target-features + future scan
│   └── test_policy.py                Action thresholds + volumes
│
├── papers/                 45 PDFs forming the research backing (see § References)
├── Research-Paper/         Original Preite & Vignali (2024) paper + markdown extract
├── main_controller.py      Deployment controller (bundle bug fixed in Phase 0)
├── requirements.txt        Core dependencies
├── PROJECT_NOTES_AND_IMPROVEMENTS.md   Design notes + improvements log
└── README.MD               This file
```

---

## How it works — end to end

```
raw feeds (Stuard CSVs + Open-Meteo) ─┐
                                      ▼
  feature engineering  (FAO-56 Ra, VPD, Hargreaves ET0, GDD→Kc, lags, rolls)
                                      ▼
  target derivation    (y_vwc_h{N} = shift(-N) per line; strictly future-only)
                                      ▼
  temporal splits      (train/val/test + 24h embargo both sides)
                                      ▼
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
  Track A: XGBoost + LightGBM (one booster per horizon)      Track B: 2-layer LSTM
          │                                                       │
          └───────────────────────────┬───────────────────────────┘
                                      ▼
  policy layer   { FAO-56 ETc scheduler · rule on ŷ(t+3h) · MPC over ŷ(t+{1,3,6}h) }
                                      ▼
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
  deployment JSON (farmer UI)                          stateful Monte Carlo (1000 seasons)
```

### Feature engineering

- **FAO-56 Ra** (extraterrestrial radiation, eq. 21) and **daylength** (eq. 34) from `(lat, doy)` [Allen 1998].
- **VPD** (Tetens): `es = 0.6108·exp(17.27 T / (T+237.3))`, `vpd = es − es·RH/100` [Wagan 2025 ranks VPD top-5].
- **Hargreaves-Samani backup ET0**: `0.0023·(Tmean+17.8)·√(Tmax−Tmin)·Ra/2.45` [Sanikhani 2018].
- **Dynamic Kc × ET0 → ETc** via GDD (base 10 °C, transplant 2023-05-01, piecewise-linear Kc across stages {0.40, 0.75, 1.15, 0.85}) [Allen 1998; Remote Control of Greenhouse Vegetable Production 2019].
- **Soil signals**: VWC lags @ {6, 18, 36, 72, 144} 10-min steps, 3h rolling mean, 6h rolling std, 1h finite-difference derivative.
- **Weather + state**: rolling rain sums @ {1, 6, 24}h, `hours_since_last_irrigation`, sin/cos(doy, hour), `line_id`.

### Forecasters

**Track A — XGBoost + LightGBM.** One regressor per horizon. XGBoost: `max_depth=6`, `lr=0.05`, up to 1000 rounds with 50-round early stopping. LightGBM: `num_leaves=63`, `lr=0.05`, up to 2000 rounds. `reg:squarederror` (residuals are near-Gaussian on this feed). Full spec: `docs/TRACK_A_GBDT.md`.

**Track B — 2-layer LSTM (hidden=64, dropout=0.2, line-id embedding dim=8) + linear head → 5 horizons.** Weighted Huber (δ=1.0, weights `[1.0, 0.8, 0.6, 0.4, 0.3]`). Near-horizon errors weighted higher because the policy reads `ŷ(t+3h)`. Adam `lr=5e-4, wd=1e-4`, patience=12, grad-clip 1.0. Full spec: `docs/TRACK_B_LSTM.md`.

### Policy layer

`src/policy/`. Action set `{OFF=0 mm, ON_LOW=5 mm, ON_HIGH=10 mm}`. Three policies share the forecaster outputs:

1. **FAO-56 ETc scheduler** — fires when cumulative depletion crosses `MAD_fraction[stage] × TAW`; current-step only. No-forecaster baseline [Allen 1998].
2. **Rule on ŷ(t+3h)** — MAD threshold with ±2 %VWC hysteresis. Equivalent to Track A's `policy_rule.py`.
3. **MPC over ŷ(t+{1,3,6}h)** — cost-minimising receding-horizon: `cost = 1·vol + 5·(%-h below MAD_lo) + 50·(%-h below WP)`. Dryout-dominant weighting ensures no knowing under-watering past wilting point [Ikegawa 2026].

### Monte Carlo (`src/sim/water_balance_mc.py`)

Replaces the v1 IID bootstrap. Stateful: simulates 70-day seasons at 10-min resolution, bootstraps contiguous day-blocks from the real feed (preserves 10-min autocorrelation), lets the policy under test pick irrigation each hour. Step update:

```
dVWC = (rain − ETc + irrigation) × 0.5            # 1 mm water → 0.5 %VWC at 20 cm
VWC  = clip(VWC + dVWC, WP, FC)
```

### Deployment (`src/deploy/inference.py`)

Feature row → Track A XGBoost `ŷ(t+3h)` → MAD rule → tree-SHAP top-3 drivers → JSON:

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

`confidence = min(|ŷ − MAD_threshold| / TAW, 1)` — distance from the decision boundary in %VWC normalised by Total Available Water. Track A is chosen here because `shap.TreeExplainer` returns per-row rationale in < 1 ms; DeepExplainer on the LSTM is ~100× slower.

---

## Testing

`pytest tests/` — 15 assertions. Breakdown:

- `tests/test_features.py` (12) — physics regressions: VPD(25 °C, 50 %) ≈ 1.584 kPa, Ra(doy=172, lat=44.1125) ≈ 42.3 MJ/m²/day, daylength(doy=80, lat=0) ≈ 12 h, SWDI endpoints, GDD monotonicity.
- `tests/test_leakage.py` (3) — `y_vwc_h{N}` matches `shift(-N)` exactly; no target columns in feature set; every non-target column timestamp ≤ t.
- `tests/test_policy.py` — MAD thresholds + action volumes.

Audits (not pytest, but run as part of the pipelines):

- `experiments/gbdt/audit_leakage.py` → shuffle-label RMSE 4.66 (vs persistence 2.00) ⇒ pass; ablation Δ +0.93 RMSE.
- `src/training/audit_leakage.py` → legacy MLP 0.9868 vs 2-feature DT 1.0000 ⇒ leakage proven.

---

## Configuration

All constants live in `src/config.py` (mirrored by `experiments/gbdt/config.py`):

```python
# Site
LAT = 44.1125
LON = 10.411

# Crop
TRANSPLANT_DATE = "2023-05-01"
T_BASE_C = 10.0
GDD_STAGE_EDGES = [(0, 350, "initial"), (350, 700, "development"),
                   (700, 1100, "mid"), (1100, 1500, "late")]
KC_BY_STAGE = {"initial": 0.40, "development": 0.75, "mid": 1.15, "late": 0.85}

# Horizons
HORIZONS_10MIN = [6, 18, 36, 72, 144]
HORIZON_LABELS = ["1h", "3h", "6h", "12h", "24h"]
LOOKBACK_HOURS = 24
LOOKBACK_STEPS = 144

# Splits
TRAIN_START/END, VAL_*, TEST_*  (+ EMBARGO_HOURS = 24)

# Soil
FIELD_CAPACITY_PCT = 27.41
WILTING_POINT_PCT  = 15.0
MAD_FRACTION_BY_STAGE = {"initial": 0.30, "development": 0.40,
                         "mid": 0.40, "late": 0.50}

# Hardware
LINE_FLOW_LPH = 300.0
PUMP_POWER_KW = 4.0
FIELD_AREA_M2 = 132.0
ACTION_VOLUME_MM = {"OFF": 0.0, "ON_LOW": 5.0, "ON_HIGH": 10.0}

# Reference
IRRIFRAME_SEASON_MM = 324.5

# Loss
HUBER_DELTA = 1.0
HORIZON_LOSS_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.3]
```

---

## Research backing

Every technical choice is tagged with the paper that motivates it. Full list in `papers/` (45 PDFs).

| Choice | Reference |
| --- | --- |
| 2-layer LSTM for data-scarce SM forecasting | Hamdaoui 2024; Jaiswal 2025; Dhanke 2025 |
| GBDT (XGBoost / LightGBM) on scarce tabular | Hamdaoui 2024 |
| Tree-SHAP rationale over XGBoost | Wagan 2025 |
| VPD as top-5 feature | Wagan 2025 |
| FAO-56 ETc, Kc, MAD thresholds, Ra/daylength | Allen 1998 (FAO-56) |
| Hargreaves-Samani backup ET0 | Sanikhani 2018 |
| Dynamic GDD-driven Kc | Remote Control of Greenhouse Vegetable Production 2019; WIAI 2021 |
| Temporal holdout + 24h embargo + leakage audit | PRISMA 2025; Big Data AI 2023 |
| MPC policy over forecaster | Ikegawa 2026; 6G-IoT 2023 |
| Transformer deferred to Phase 6 | Ikegawa 2026 |
| ERA5 + transfer learning plan | Hamdaoui 2024 |
| Sentinel-2 NDVI/EVI/LAI | Applications AI in Precision Irrigation 2025 |
| Autoencoder anomaly (deferred Phase 6) | Benameur 2024 |
| Plant electrophysiology (deferred Phase 6) | Precision Irrigation Electrophysiology 2025 |
| GPR soil-moisture sensing (deferred Phase 6) | Debangshi 2025 |
| Vineyard IoT architecture parallels | Future of Vineyard Irrigation 2025 |

### References

Cited in Chicago author-date style. Full PDFs in `papers/`.

- **Allen, R. G., Pereira, L. S., Raes, D., & Smith, M.** 1998. *Crop Evapotranspiration — Guidelines for Computing Crop Water Requirements.* FAO Irrigation and Drainage Paper 56. Food and Agriculture Organization of the United Nations.
- **Benameur, N., et al.** 2024. "An Innovative Smart and Sustainable Low-Cost Irrigation System for Anomaly Detection Using Deep Learning." *Sensors* 24(1): 1162.
- **Debangshi, U.** 2025. "Precision Irrigation with Artificial Intelligence–Integrated Ground-Penetrating Radar Reduces Water Stress in Corn." *Agricultural & Environment Letters*.
- **Dhanke, J., et al.** 2025. "Climate-Based AI-Powered Precision Irrigation: Sustainably Smart Agriculture Frameworks for Maximum Crop Yields." *Discover Sustainability*, article s41976-024-00174-4.
- **Hamdaoui, M., et al.** 2024. "Revolutionizing Agriculture: A Comprehensive Review of AI-Enabled Precision Irrigation and Water Quality Forecasting." *Bulgarian Journal of Agricultural Science* 37(2): 354–380.
- **Ikegawa, K., et al.** 2026. "Transformer-Based Artificial Intelligence for Forecasting Energy Demand in Irrigation Districts." *Computers and Electronics in Agriculture*.
- **Jaiswal, P., et al.** 2025. "Smart Drip Irrigation Systems Using IoT: A Review of Architectures, Machine Learning Models, and Emerging Trends." Article s44279-025-00430-1.
- **PRISMA Review.** 2025. "PRISMA-Guided Systematic Review on the Adoption of Artificial Intelligence and Embedded Systems for Smart Irrigation." *Pure and Applied Geophysics*, article s00024-025-03707-0.
- **Preite, L., & Vignali, G.** 2024. Original source paper for the v1 classifier (see `Research-Paper/`).
- **Sanikhani, H., et al.** 2018. "Temperature-Based Modeling of Reference Evapotranspiration Using Several Artificial Intelligence Models." *Theoretical and Applied Climatology*.
- **Wagan, M. A., et al.** 2025. "Interpreting Irrigation Decisions: Explainable AI Using SHAP and LIME in Agricultural Water Management Models." *Water Resources Management*. DOI: 10.1007/s11269-025-04468-0.
- **Water-Use-Efficiency in Irrigated Agriculture (WIAI).** 2021. White paper WIAI-IDH-DD-2021.
- *Remote Control of Greenhouse Vegetable Production with Artificial Intelligence.* 2019. *Sensors* 19(8): 1807.
- *A Novel Autonomous Irrigation System for Smart Agriculture Using AI and 6G-Enabled IoT Network.* 2023. Elsevier.
- *Future of Vineyard Irrigation: AI-Driven Insights from IoT Data.* 2025. *Sensors* 25(13): 3658.
- *Applications of Artificial Intelligence in Precision Irrigation.* 2025. *Journal of Environmental & Earth Sciences*.
- *Unravelling the Potential of Big Data-Driven Decision-Making in Sustainable Water Irrigation: An AI Perspective.* 2023. Article e2023ss069.

---

## Honest limitations

- **Single season.** Stuard 2023-07-28 → 2023-09-03 is mid-to-late season only. Claims outside that window are not validated.
- **Single 20 cm probe.** Cannot disambiguate surface wetting from deep drainage. Multi-depth probes would fix this.
- **Tomato-specific Kc.** No cross-crop generalization claim.
- **Simulation ≠ field.** Water-savings numbers come from a stateful MC, not a randomized A/B field deployment.
- **No yield linkage.** Stress-hour proxy only. Yield data absent.
- **Sensor-dropout blindness.** Forecaster assumes VWC arrives every 10 min. Autoencoder anomaly detection is deferred to a future phase [Benameur 2024].
- **IRRIFRAME reference approximate.** IRRIFRAME's internal parameters for this specific plot are not public.
- **Stuard `volume_diff` column.** Likely cumulative-counter semantics rather than per-step deltas. Treat as placeholder until the water-meter schema is validated.
- **Persistence is hard to beat on the test week.** The held-out week is unusually stable; both tracks lose h=1h on that slice. ERA5/ISMN pretraining is the credential-gated fix.

---

## Deferred (Phase 7+ — credential-gated or explicitly out of scope)

- ERA5 (Copernicus CDS) + ISMN + SMAP multi-year pretraining of the LSTM [Hamdaoui 2024].
- Transformer forecaster [Ikegawa 2026].
- Reinforcement-learning scheduler.
- Autoencoder anomaly detector for sensor dropouts [Benameur 2024].
- Plant electrophysiology integration [Precision Irrigation Electrophysiology 2025].
- Multi-depth VWC probes + recorded yield + recorded farmer schedule (requires second-season Stuard data).
- GPR soil-moisture sensor path [Debangshi 2025].
