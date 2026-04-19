# Track A — GBDT forecaster + rule policy

Forecast target: future soil VWC. No threshold-derived label; see plan.

## Forecast metrics (RMSE | MAE | NSE)

### val

| model | 1h | 3h | 6h | 12h | 24h |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 2.00 / 0.54 / 0.80 | 3.80 / 1.49 / 0.27 | 4.87 / 2.46 / -0.23 | 5.06 / 3.24 / -0.63 | 5.71 / 4.37 / -1.43 |
| climatology | 4.41 / 3.07 / 0.03 | 4.38 / 3.04 / 0.02 | 4.35 / 2.98 / 0.02 | 3.73 / 2.61 / 0.11 | 3.42 / 2.26 / 0.13 |
| water_balance | 1.99 / 0.54 / 0.80 | 3.75 / 1.46 / 0.29 | 4.75 / 2.44 / -0.18 | 5.03 / 3.47 / -0.62 | 7.31 / 5.15 / -2.98 |
| xgboost | 1.97 / 0.82 / 0.81 | 3.74 / 1.85 / 0.29 | 3.93 / 2.15 / 0.19 | 3.31 / 2.14 / 0.30 | 3.70 / 2.12 / -0.02 |
| lightgbm | 1.70 / 0.66 / 0.86 | 3.50 / 1.85 / 0.38 | 3.88 / 1.99 / 0.22 | 3.34 / 2.02 / 0.29 | 3.68 / 2.35 / -0.01 |

### test

| model | 1h | 3h | 6h | 12h | 24h |
| --- | ---: | ---: | ---: | ---: | ---: |
| persistence | 0.83 / 0.16 / 0.94 | 1.38 / 0.41 / 0.83 | 1.62 / 0.64 / 0.75 | 1.63 / 0.95 / 0.71 | 2.11 / 1.64 / 0.43 |
| climatology | 3.39 / 2.53 / -0.04 | 3.41 / 2.55 / -0.05 | 3.27 / 2.46 / -0.02 | 2.90 / 2.23 / 0.09 | 2.38 / 1.87 / 0.27 |
| water_balance | 1.00 / 0.30 / 0.91 | 2.16 / 0.82 / 0.58 | 3.70 / 1.51 / -0.30 | 6.92 / 2.68 / -4.15 | 12.04 / 4.74 / -17.57 |
| xgboost | 1.39 / 0.81 / 0.82 | 3.25 / 2.25 / 0.04 | 2.96 / 2.30 / 0.17 | 2.87 / 2.02 / 0.11 | 2.58 / 2.12 / 0.15 |
| lightgbm | 0.99 / 0.48 / 0.91 | 2.86 / 2.16 / 0.26 | 3.43 / 2.88 / -0.12 | 3.37 / 2.86 / -0.23 | 2.97 / 2.34 / -0.13 |

## Policy backtest (mm of water per line)

| split/model | days | lines | policy mm/line | policy mm/line/day | Stuard mm/line | action counts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| xgboost_val | 4.99 | 3 | 271.67 | 54.409 | 3308.33 | {'OFF': 251, 'ON_LOW': 55, 'ON_HIGH': 54} |
| xgboost_test | 5.99 | 3 | 8.33 | 1.39 | 0.0 | {'OFF': 427, 'ON_LOW': 5} |
| lightgbm_val | 4.99 | 3 | 268.33 | 53.741 | 3308.33 | {'OFF': 243, 'ON_LOW': 73, 'ON_HIGH': 44} |
| lightgbm_test | 5.99 | 3 | 96.67 | 16.13 | 0.0 | {'OFF': 376, 'ON_LOW': 54, 'ON_HIGH': 2} |

IRRIFRAME full-season reference: **324.5 mm**.

## Notes
- Backing papers: Hamdaoui 2024 (tree ensembles on scarce tabular); Wagan 2025 (SHAP + XGB).
- This report feeds Phase 6 cross-track comparison vs Track B LSTM.