# Physics-based Monte Carlo (Phase 4)

Seasons simulated: **1000** · season length 70 days · seed 0.
Weather bootstrapped from `data/processed/modeling_dataset_v2.parquet` (day-block).
IRRIFRAME reference: **324.5 mm** per plot, full 2023 season.

| policy | mm/season (mean) | 95% CI | stress h<WP (mean) | deficit %·h<MAD (mean) | savings vs IRRIFRAME (mean %) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fao56 | 303.3 | [258.2, 348.4] | 0.0 | 0.0 | +6.5 |
| rule_persistence | 104.4 | [68.5, 140.3] | 0.0 | 0.0 | +67.8 |

## Honest reading
- `rule_persistence` uses `vwc_hat(t+3h) = vwc(t)` as its forecast; no learned model runs inside the MC. Coupling the LSTM inside the MC is Phase 6 work.
- `savings_vs_irriframe_pct` can go negative (over-watering) or close to 100% (under-watering causing crop stress). Read alongside the stress/deficit columns — saving water while starving the crop is not a win.
- Unlike the legacy v1 MC, state is preserved across steps and the VWC trajectory responds to rain and ETc. Legacy numbers were IID samples; these are not comparable.