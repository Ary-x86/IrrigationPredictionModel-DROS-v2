"""MPC policy over the LSTM's 6h-ahead forecast. Phase-3b.

Receding-horizon control:
  At each decision step for each line, enumerate actions
  {OFF, ON_LOW=5mm, ON_HIGH=10mm}. For each action, simulate the resulting
  VWC trajectory over the next 6h using:
    - forecasted baseline vwc(t+h) from the LSTM
    - + action-induced wetting bump (volume_mm / root_depth_mm)
  Compute cost:
    cost = water_used + stress_penalty * cumulative_hours_below_MAD_lo
         + dryout_penalty * cumulative_hours_below_WP
  Pick the action with minimum cost.

Simplification: we treat the wetting bump as instantaneous and uniform over
the horizon — the full MPC substitutes in the forecaster's delta under each
action; Track B defers that coupling to Phase 6 when both tracks are compared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ACTION_VOLUME_MM, WILTING_POINT_PCT
from src.policy.policy_rule import mad_threshold

ROOT_DEPTH_MM = 200.0  # 20 cm → volumetric conversion: 1 mm water ≈ 0.5 VWC%
WETTING_VWC_PER_MM = 100.0 / ROOT_DEPTH_MM  # % VWC added per mm applied at 20cm


@dataclass
class MPCParams:
    water_weight: float = 1.0           # cost per mm applied
    stress_weight: float = 5.0          # cost per %-hour below MAD_lo
    dryout_weight: float = 50.0         # cost per %-hour below WP
    horizon_hours: tuple[int, ...] = (1, 3, 6)


def _cost(vwc_traj_pct: np.ndarray, volume_mm: float, stage: str, params: MPCParams) -> float:
    mad_lo = mad_threshold(stage) - 2.0
    below_mad = np.clip(mad_lo - vwc_traj_pct, 0.0, None).sum()
    below_wp = np.clip(WILTING_POINT_PCT - vwc_traj_pct, 0.0, None).sum()
    return (
        params.water_weight * volume_mm
        + params.stress_weight * float(below_mad)
        + params.dryout_weight * float(below_wp)
    )


def decide(
    forecast_map: dict[str, np.ndarray],
    stages: pd.Series,
    params: MPCParams | None = None,
) -> pd.DataFrame:
    """Return one action per row given per-horizon forecast arrays.

    forecast_map keys must include '1h', '3h', '6h'. Arrays are aligned to
    `stages` by index.
    """
    params = params or MPCParams()
    n = len(stages)
    actions = np.full(n, "OFF", dtype=object)
    volumes = np.zeros(n, dtype=float)
    costs = np.zeros((n, 3), dtype=float)

    base_1h = forecast_map["1h"]
    base_3h = forecast_map["3h"]
    base_6h = forecast_map["6h"]

    for i in range(n):
        stage = str(stages.iloc[i])
        best_cost = np.inf
        best_action = "OFF"
        best_volume = 0.0
        for j, (action, vol_mm) in enumerate(ACTION_VOLUME_MM.items()):
            bump = vol_mm * WETTING_VWC_PER_MM
            traj = np.array([base_1h[i] + bump, base_3h[i] + bump, base_6h[i] + bump])
            c = _cost(traj, vol_mm, stage, params)
            costs[i, j] = c
            if c < best_cost:
                best_cost = c
                best_action = action
                best_volume = vol_mm
        actions[i] = best_action
        volumes[i] = best_volume

    out = pd.DataFrame({
        "action": actions,
        "volume_mm": volumes,
        "cost_off": costs[:, 0],
        "cost_on_low": costs[:, 1],
        "cost_on_high": costs[:, 2],
    })
    return out


def backtest(
    df: pd.DataFrame,
    forecast_map: dict[str, np.ndarray],
    params: MPCParams | None = None,
) -> dict:
    """Run `decide` on a split_df, aggregate per-line water use and actions."""
    decisions = decide(forecast_map, df["growth_stage"].reset_index(drop=True), params)
    joined = df.reset_index(drop=True).assign(
        _action=decisions["action"].values,
        _volume=decisions["volume_mm"].values,
    )
    joined["_hour"] = joined["datetime"].dt.floor("h")
    hourly = joined.drop_duplicates(subset=["line", "_hour"], keep="first")

    total_mm = float(hourly["_volume"].sum())
    n_lines = int(hourly["line"].nunique())
    per_line_mm = total_mm / max(n_lines, 1)
    days = max((joined["datetime"].max() - joined["datetime"].min()).total_seconds() / 86400.0, 1e-6)
    stuard_mm = float(df["volume_diff"].fillna(0.0).sum()) / max(n_lines, 1)
    return {
        "days": round(days, 2),
        "n_lines": n_lines,
        "policy_mm_per_line": round(per_line_mm, 2),
        "policy_mm_per_line_per_day": round(per_line_mm / days, 3),
        "stuard_mm_per_line": round(stuard_mm, 2),
        "action_counts": hourly["_action"].value_counts().to_dict(),
    }
