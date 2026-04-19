# Track A leakage audit

Target: `y_vwc_h1h`  ·  elapsed 71.9s

## Shuffle-label sanity
- shuffled val RMSE: **4.663**
- persistence val RMSE: **2.002**
- pass (shuffled >= persistence): **True**

## Feature ablation (drop vwc_20cm + lags)
- full-feature val RMSE: **1.877**
- ablated val RMSE: **2.809**
- degradation: **+0.932**
- dropped: vwc_20cm, vwc_20cm_deriv_1h, vwc_20cm_lag_144s, vwc_20cm_lag_18s, vwc_20cm_lag_36s, vwc_20cm_lag_6s, vwc_20cm_lag_72s, vwc_20cm_roll_mean_3h, vwc_20cm_roll_std_6h

## Permutation importance (val RMSE - base)

| feature | delta_rmse |
| --- | ---: |
| vwc_20cm | +3.0408 |
| hours_since_last_irrigation | +0.5427 |
| vwc_20cm_deriv_1h | +0.0583 |
| air_temp_c | +0.0266 |
| vwc_20cm_lag_6s | +0.0248 |
| line_id | +0.0224 |
| vpd_kpa | +0.0041 |
| rh_pct | +0.0040 |
| et0_hargreaves_mm_day | +0.0029 |
| et0_open_meteo_mm_h | +0.0012 |
| vwc_20cm_lag_144s | +0.0003 |
| rain_mm_h | +0.0000 |
| rain_mm_1h_sum | +0.0000 |
| rain_mm_6h_sum | +0.0000 |
| rain_mm_24h_sum | +0.0000 |
| ra_mj_m2_day | +0.0000 |
| daylength_h | +0.0000 |
| gdd_cum | +0.0000 |
| kc_dynamic | +0.0000 |
| sin_doy | +0.0000 |
| cos_doy | +0.0000 |
| vwc_20cm_roll_mean_3h | -0.0018 |
| etc_mm_h | -0.0066 |
| sin_hour | -0.0148 |
| soil_temp_c | -0.0150 |
| cos_hour | -0.0170 |
| vwc_20cm_lag_72s | -0.0188 |
| vwc_20cm_lag_18s | -0.0290 |
| vwc_20cm_lag_36s | -0.0437 |
| vwc_20cm_roll_std_6h | -0.0802 |