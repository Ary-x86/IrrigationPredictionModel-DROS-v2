# Track B policy backtest

Policies:
- `fao56`: Allen FAO-56 ETc-threshold scheduler (current-step only).
- `rule_lstm_3h`: rule policy on the LSTM's t+3h forecast.
- `mpc_lstm_6h`: MPC optimizing cost over the LSTM's 1h/3h/6h trajectory.

IRRIFRAME reference (full 2023 season): **324.5 mm** per plot.

## val

| policy | days | lines | mm/line | mm/line/day | Stuard mm/line | actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fao56 | 4.99 | 3 | 321.67 | 64.423 | 3308.33 | {'OFF': 252, 'ON_HIGH': 85, 'ON_LOW': 23} |
| rule_lstm_3h | 4.99 | 3 | 466.67 | 93.463 | 3308.33 | {'OFF': 159, 'ON_LOW': 122, 'ON_HIGH': 79} |
| mpc_lstm_6h | 4.99 | 3 | 93.33 | 18.693 | 3308.33 | {'OFF': 304, 'ON_LOW': 56} |

## test

| policy | days | lines | mm/line | mm/line/day | Stuard mm/line | actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fao56 | 5.99 | 3 | 721.67 | 120.417 | 0.0 | {'ON_HIGH': 207, 'OFF': 206, 'ON_LOW': 19} |
| rule_lstm_3h | 5.99 | 3 | 820.0 | 136.825 | 0.0 | {'ON_HIGH': 244, 'OFF': 184, 'ON_LOW': 4} |
| mpc_lstm_6h | 5.99 | 3 | 406.67 | 67.856 | 0.0 | {'ON_LOW': 244, 'OFF': 188} |

## Caveats
- `stuard_mm_per_line` reads `volume_diff` from the raw sensor feed; on this feed it likely reflects cumulative-counter semantics, not per-step deltas. Treat as a placeholder column until the water-meter schema is validated.
- MPC wetting bump uses a fixed `WETTING_VWC_PER_MM = 0.5` (20cm root depth). Phase 6 will couple the bump to the forecaster's residual-under-action.