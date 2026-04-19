# SHAP report (Track A XGBoost h=3h)

Samples: **20** from test split. Model: `models/xgb_h3h.json`.
Backing: Wagan 2025 — tree-SHAP for per-decision irrigation rationale.

## Mean |SHAP| across sampled decisions (top 10)

| feature | mean |SHAP| |
| --- | ---: |
| vwc_20cm | 1.3974 |
| hours_since_last_irrigation | 0.9597 |
| etc_mm_h | 0.3332 |
| line_id | 0.2734 |
| air_temp_c | 0.2201 |
| vwc_20cm_lag_6s | 0.1626 |
| soil_temp_c | 0.1242 |
| vwc_20cm_lag_36s | 0.1213 |
| vwc_20cm_roll_std_6h | 0.1210 |
| ra_mj_m2_day | 0.1200 |

## Per-decision top-3 drivers (first 5)

| datetime | line | pred VWC 3h | top feature | 2nd | 3rd |
| --- | ---: | ---: | --- | --- | --- |
| 2023-08-29 07:00:00 | 0 | 29.81 | vwc_20cm (+5.57) | vwc_20cm_lag_6s (+0.57) | line_id (+0.36) |
| 2023-08-29 17:30:00 | 0 | 28.64 | vwc_20cm (+4.74) | vwc_20cm_lag_6s (+0.49) | line_id (+0.32) |
| 2023-08-30 08:20:00 | 0 | 28.95 | vwc_20cm (+4.76) | vwc_20cm_lag_6s (+0.48) | line_id (+0.31) |
| 2023-09-01 03:20:00 | 0 | 25.47 | vwc_20cm (+2.00) | ra_mj_m2_day (-0.24) | vwc_20cm_lag_6s (+0.21) |
| 2023-09-02 19:40:00 | 0 | 24.22 | vwc_20cm (+1.01) | hours_since_last_irrigation (+0.19) | vwc_20cm_roll_mean_3h (+0.15) |

## Notes
- Base value (expected prediction on train) is added to the SHAP sum per row.
- `top3_drivers` feeds deployment's `reason_top3_features`.
- LSTM DeepExplainer is deferred (~100x slower, not needed for farmer UI).