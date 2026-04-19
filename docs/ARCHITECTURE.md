# End-to-End Architecture — Non-DROS v2

Map of how raw sensor bytes become a farmer-readable irrigation decision, and
where each paper in `papers/` shows up in code.

> Two forecasters (Track A GBDT + Track B LSTM) sit behind a single policy
> layer. Physics features are shared. A stateful Monte Carlo exists to
> estimate water-use envelopes beyond the 37-day Stuard slice.

---

## 1. Why we rebuilt (the leakage story)

The v1 classifier was reported at ~99 % accuracy because its *label*
(`Irrigation_Decision`) was computed deterministically from two of its six
input features inside `src/03_soil_capacity_calculator.py`. The MLP was
re-learning a hand-written threshold rule, then the evaluation compared the
learned rule to itself. `reports/leakage_audit_phase0.md` documents this
empirically — a 2-feature decision tree fits the target perfectly.

**Fix:** flip the target from "class derived from current features" to
"future VWC at h ∈ {1h, 3h, 6h, 12h, 24h}", then derive the irrigation
action from the forecast. Because the target is the physically-future
value, no current-step feature can be a deterministic function of it. The
leakage route is *structurally* closed, not just audited.

---

## 2. Data flow

```
raw sensor feeds + Open-Meteo
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Feature engineering (src/features/*.py, experiments/gbdt/   │
│  build_features.py)                                          │
│  - FAO-56 Ra, daylength, Hargreaves ET0                      │
│  - VPD (Tetens)                                              │
│  - GDD → growth stage → dynamic Kc → ETc                     │
│  - VWC lags/rolls/derivatives                                │
│  - rain rolling sums, irrigation clock                       │
│  - sin/cos doy + hour, line_id                               │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Target derivation (src/labels/derive_targets.py)            │
│  y_vwc_h{N} = vwc_20cm.shift(-N) per line                    │
│  Strictly future-only, nothing else.                         │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Temporal splits with 24h embargo                            │
│  (src/splits/temporal_split.py,                              │
│   experiments/gbdt/temporal_split.py)                        │
└──────────────────────────────────────────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
Track A    Track B
(GBDT)     (LSTM)
XGBoost +  2-layer
LightGBM   LSTM+head
   │         │
   └────┬────┘
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Policy layer (src/policy/)                                  │
│  - policy_fao56: ETc-threshold scheduler                     │
│  - policy_rule:  MAD rule on ŷ(t+3h)                         │
│  - policy_mpc:   cost-min over {OFF, ON_LOW, ON_HIGH}        │
│                  using ŷ at h=1h, 3h, 6h                     │
└──────────────────────────────────────────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
Deployment  Simulation
JSON        (src/sim/water_balance_mc.py)
(src/deploy)
```

---

## 3. Shared physics (src/features/)

| Module | What | Formula / source |
| --- | --- | --- |
| `time_features.py` | `sin/cos_{doy,hour}` | periodic encodings |
| `radiation.py` | `ra_mj_m2_day`, `daylength_h` | FAO-56 eq. 21 & 34 |
| `et.py` | `vpd_kpa`, `et0_hargreaves_mm_day`, `etc_mm_h` | Tetens; Sanikhani 2018 |
| `phenology.py` | `gdd_cum`, `growth_stage`, `kc_dynamic` | base 10 °C from 2023-05-01 transplant; Kc piecewise-linear |
| `soil.py` | VWC lags, rolls, `vwc_20cm_deriv_1h`, `swdi` | standard signal features |
| `weather.py` | rain rolling sums, `hours_since_last_irrigation` | bookkeeping |

Track A reimplements these formulas locally in `experiments/gbdt/
build_features.py` so it has zero `src/` dependency (per plan: isolated
prototype that ships first).

**Verification:** `tests/test_features.py` runs 12 known-answer checks. The
canonical ones:
- VPD(T=25 °C, RH=50 %) ≈ 1.584 kPa
- Ra(doy=172, lat=44.1125) ≈ 42.3 MJ/m²/day (summer solstice)
- Daylength(doy=80, lat=0) ≈ 12 h (equinox at equator)
- SWDI at FC = 1.0, at WP = 0.0

---

## 4. Forecasters (side by side)

| Aspect | Track A GBDT | Track B LSTM |
| --- | --- | --- |
| Library | XGBoost + LightGBM | PyTorch |
| Input | 30 flat features per row | 144-step × 18-feature sequence + line_id |
| Output | one scalar per booster, 5 boosters | 5-dim vector, one model |
| Loss | squared error | weighted Huber (δ=1.0, w=[1, 0.8, 0.6, 0.4, 0.3]) |
| Training | ~90 s total CPU | ~242 s CPU (17 epochs early-stopped) |
| Disk | 5.03 MB (10 boosters) | 0.23 MB (one .pt) |
| Inference | 4.2 ms / 1000 preds (one booster) | 342 ms / 1000 preds (all 5 horizons) |
| SHAP | native `TreeExplainer` | permutation-importance proxy |

Full docs: `docs/TRACK_A_GBDT.md`, `docs/TRACK_B_LSTM.md`.
Head-to-head results: `reports/track_comparison.md`.

**Why both exist:** user preference saved in memory — *"prefer running
competing approaches in parallel, compare empirically."* Track A is the
shippable path today; Track B is where the ERA5-pretraining upside lives.

---

## 5. Policy layer

Same layer for both tracks. Action set `{OFF=0, ON_LOW=5 mm, ON_HIGH=10 mm}`
from `src/config.py:52`.

### 5.1 FAO-56 ETc-threshold (`src/policy/policy_fao56.py`)

Tracks cumulative soil-moisture depletion `depleted_mm` and fires when
`depleted_mm ≥ MAD_fraction × TAW`. Uses current-step features only — the
"no-forecaster" baseline every ML decision must beat.

Backing: Allen FAO-56 + *Water-Use-Efficiency-in-Irrigated-Agriculture-web.pdf*.

### 5.2 Rule on forecasted VWC (`src/policy/policy_rule.py`)

```python
mad_hi = FC − MAD_fraction[stage] × TAW
mad_lo = mad_hi − 2.0                   # %VWC hysteresis band

if ŷ(t+3h) < mad_lo:  action = ON_HIGH  (10 mm)
elif ŷ(t+3h) < mad_hi: action = ON_LOW  (5 mm)
else:                 action = OFF
```

`MAD_fraction_by_stage = {initial: 0.30, development: 0.40, mid: 0.40,
late: 0.50}` (tomato, FAO-56 §3.7).

With `FC=27.41 %, WP=15.0 %, TAW=12.41 %`, the development-stage threshold
is `27.41 − 0.40 × 12.41 = 22.45 %`.

### 5.3 Model-Predictive Control (`src/policy/policy_mpc.py`)

Receding-horizon over 3 actions using ŷ at h∈{1,3,6}h and a wetting bump
`Δvwc = volume_mm × 0.5` (20cm root depth). Cost:

```
cost = water_weight · volume_mm
     + stress_weight · Σ max(mad_lo − ŷ, 0)   # %-hours below MAD
     + dryout_weight · Σ max(WP    − ŷ, 0)    # %-hours below WP
```

Weights from `MPCParams`: `{water: 1.0, stress: 5.0, dryout: 50.0}`. The
dryout penalty dominates so the MPC will never knowingly under-water past WP.

Backing: Ikegawa 2026 (MPC over forecaster); *Novel autonomous irrigation
6G-IoT*.

### 5.4 Policy backtest on held-out data (`reports/policy_backtest.md`)

| split | policy | mm/line | mm/line/day | action mix |
| --- | --- | ---: | ---: | --- |
| val | fao56 | 321.67 | 64.42 | OFF 252 / ON_HIGH 85 / ON_LOW 23 |
| val | rule_lstm_3h | 466.67 | 93.46 | OFF 159 / ON_LOW 122 / ON_HIGH 79 |
| val | mpc_lstm_6h | **93.33** | **18.69** | OFF 304 / ON_LOW 56 |
| test | fao56 | 721.67 | 120.42 | ON_HIGH 207 / OFF 206 / ON_LOW 19 |
| test | mpc_lstm_6h | **406.67** | 67.86 | ON_LOW 244 / OFF 188 |

MPC is dramatically water-light because the dryout penalty is only paid
when ŷ crosses WP, and on this feed the forecaster never predicts that.

---

## 6. Monte Carlo (`src/sim/water_balance_mc.py`)

Replaces the v1 IID bootstrap (which sampled rows independently with no
state). The new version is **stateful**: it simulates a 70-day tomato season
at 10-min resolution, preserves soil VWC across steps, bootstraps contiguous
day-blocks from the real feed to keep the 10-min autocorrelation structure,
and lets the policy under test pick irrigation each hour.

Step update:

```
dVWC = (rain − ETc(step) + irrigation) × WETTING_VWC_PER_MM
VWC  = clip(VWC + dVWC, WP, FC)
depleted_mm = max(depleted_mm + ETc(step) − rain − irrigation, 0)
```

`WETTING_VWC_PER_MM = 100 / 200 = 0.5` (1 mm water → 0.5 %VWC in a 20 cm
profile).

### 6.1 Results at 1000 seasons (`reports/monte_carlo.md`)

| policy | mm/season (mean) | 95 % CI | stress h | savings vs IRRIFRAME |
| --- | ---: | ---: | ---: | ---: |
| fao56 | 303.3 | [258.2, 348.4] | 0.0 | +6.5 % |
| rule_persistence | 104.4 | [68.5, 140.3] | 0.0 | +67.8 % |

`rule_persistence` assumes `ŷ(t+3h) = vwc(t)` — the naive forecast — to keep
the MC self-contained. Coupling the LSTM inside the MC is Phase 6 work and
intentionally deferred so one reviewer can understand the physics side
without touching the ML side.

### 6.2 Why these numbers should not be read as "savings"

- `savings_vs_irriframe_pct` can hit 68 % by simply under-watering into
  plant stress. On a 70-day synthetic season with zero stress hours, the
  number is meaningful. With stress hours > 0, it is not. That is why the
  report always shows stress alongside savings.
- IRRIFRAME's internal parameters for this plot aren't public, so the 324.5
  mm reference is approximate — see `reports/policy_backtest.md` § Caveats.

---

## 7. Deployment (`src/deploy/inference.py`)

Single-shot entrypoint. Accepts a feature row, emits JSON with enough
context for a farmer UI to show what the decision is and why:

```json
{
  "action": "OFF" | "ON_LOW" | "ON_HIGH",
  "volume_mm": 0.0 | 5.0 | 10.0,
  "predicted_vwc_3h": <float>,
  "mad_threshold": <float>,
  "growth_stage": "initial|development|mid|late",
  "reason_top3_features": [
    {"feature": ..., "value": ..., "shap": ..., "direction": "wetter|drier"}
    ...
  ],
  "confidence": <[0, 1]>
}
```

Forecaster behind this path: Track A XGBoost at h=3h, because `shap.
TreeExplainer` gives per-row rationale in <1 ms. The LSTM path would need
DeepExplainer (~100× slower, not farmer-facing).

`confidence = min(|ŷ − MAD| / TAW, 1)` — distance from the decision
boundary in VWC units normalized by Total Available Water. 0 at the
threshold (genuinely ambiguous), 1 at either saturation endpoint.

---

## 8. Audits + CI gates

| Guard | File | What it catches |
| --- | --- | --- |
| Target-alignment regression | `tests/test_leakage.py` | `y_vwc_h{N}` drifts from `shift(-N)` |
| No-target-features regression | `tests/test_leakage.py` | `y_vwc_h*` leaking into `SEQ_FEATURE_COLS` |
| Future-timestamp scan | `tests/test_leakage.py` | Any non-target column timestamp > t |
| Shuffle-label sanity | `experiments/gbdt/audit_leakage.py` | Model fits permuted targets (i.e. cheating) |
| Permutation importance | same | One feature explains > 0.4 R² alone |
| Feature ablation | same | Model fails without `vwc_20cm` + lags |
| Feature-formula regression | `tests/test_features.py` | VPD, Ra, daylength, SWDI drift numerically |

Current state: all 15 tests pass. `reports/leakage_audit_phase0.md`
documents the v1 leakage; `experiments/gbdt/reports/leakage_audit.md`
documents v2 passes.

---

## 9. File → purpose quick reference

```
src/config.py                          — LAT, LON, FC, WP, MAD, splits, Kc, action volumes
src/features/                          — all physics (VPD, Ra, ET0, Kc, GDD, rolls)
src/labels/derive_targets.py           — y_vwc_h* via shift(-N)
src/splits/temporal_split.py           — dates + 24h embargo
src/models/lstm_forecaster.py          — Track B LSTM
src/models/{persistence,water}.py      — baselines
src/training/
  dataset.py                           — sequence builder + stats
  train_forecaster.py                  — LSTM trainer
  audit_leakage.py                     — v1 MLP leakage proof
src/policy/
  policy_fao56.py                      — FAO-56 ETc scheduler
  policy_rule.py                       — rule on ŷ(t+3h)
  policy_mpc.py                        — MPC over ŷ(t+{1,3,6}h)
  backtest.py                          — policy backtest driver
src/sim/water_balance_mc.py            — stateful physics MC (Phase 4)
src/eval/
  metrics.py                           — RMSE, MAE, NSE
  compare_baselines.py                 — LSTM vs physics baselines
  compare_tracks.py                    — Phase 6 GBDT vs LSTM
src/explain/shap_report.py             — Wagan 2025 tree-SHAP pattern
src/deploy/inference.py                — farmer-facing JSON

experiments/gbdt/                      — isolated Track A prototype
  build_features.py                    — local reimplementation of src/features/
  train_{xgboost,lightgbm}.py          — one booster per horizon
  baselines.py                         — persistence, climatology, water-balance
  policy_rule.py                       — mirrors src/policy/policy_rule.py
  audit_leakage.py                     — shuffle + permutation + ablation
  evaluate.py                          — reports/report.md

data/raw/                              — stuard_* CSVs, ERA5 GRIBs (later), S2 rasters (later)
data/interim/merged_sensor_data.csv    — per-line 10-min grid joined with weather
data/processed/modeling_dataset_v2.parquet  — canonical feature + target table

models/forecaster_lstm.pt              — Track B checkpoint
models/legacy_mlp.pkl                  — v1 MLP kept as leakage-audit baseline
experiments/gbdt/models/*.{json,txt}   — Track A boosters

reports/
  leakage_audit_phase0.md              — why we rebuilt
  baseline_comparison.md               — LSTM vs physics baselines
  policy_backtest.md                   — rule + MPC backtest
  monte_carlo.md                       — 1000-season MC
  shap_report.md                       — Wagan tree-SHAP output
  track_comparison.md                  — Phase 6 head-to-head
  shap_examples/                       — per-decision JSON

docs/
  TRACK_A_GBDT.md                      — GBDT deep dive
  TRACK_B_LSTM.md                      — LSTM deep dive
  ARCHITECTURE.md                      — this file

tests/
  test_features.py                     — known-answer physics checks
  test_leakage.py                      — target alignment + future scan
  test_policy.py                       — action thresholds + volumes
```

---

## 10. Research backing (paper → location in code)

| Paper | Where it shows up |
| --- | --- |
| Hamdaoui 2024 (PRISMA review) | 2-layer LSTM choice, GBDT-on-scarce-tabular case |
| Jaiswal 2025 (*Smart drip IoT review*) | Multi-horizon direct regression head |
| Dhanke 2025 DLISA | 5-horizon head pattern |
| Ikegawa 2026 | MPC over forecaster; Transformer deferred to Phase 6 |
| Wagan 2025 (*SHAP + LIME*) | Tree-SHAP deployment rationale; VPD top-5 feature |
| Allen FAO-56 | Ra eq. 21, daylength eq. 34, Kc curves, MAD thresholds, depletion bucket |
| Sanikhani 2018 | Hargreaves-Samani backup ET0 |
| Water-Use-Efficiency white paper + Remote Control of Greenhouse Vegetable Production (Sensors 2019) | Dynamic GDD-driven Kc |
| *Novel autonomous irrigation 6G-IoT* | MPC architecture |
| *Future of Vineyard Irrigation 2025* | Line-level embedding rationale |
| Benameur 2024 (*sensors-24-01162*) | Autoencoder anomaly — deferred Phase 6 |
| *Precision irrigation with plant electrophysiology 2025* | Deferred Phase 6 |
| *Precision irrigation with GPR 2025* (Debangshi) | Deferred Phase 6 |

---

## 11. What this system does **not** do (honest limits)

- No cross-season generalization evidence. Stuard 2023-07-28 → 2023-09-03 only.
- No cross-crop generalization. Tomato Kc curve everywhere.
- No multi-depth root-zone modelling. Single 20 cm probe.
- No causal water-savings claim. Simulation numbers ≠ field results.
- No yield coupling. Stress hours are a proxy, not a yield-validated outcome.
- No sensor-dropout handling. Forecaster assumes VWC arrives every 10 min.
- IRRIFRAME reference (324.5 mm) is an approximation.

A second season of Stuard data with multi-depth VWC, recorded yield, and the
real farmer schedule would unlock the first five items.
