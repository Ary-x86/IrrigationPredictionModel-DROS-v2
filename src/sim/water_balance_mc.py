"""Physics-based Monte Carlo replacement for src/05_monte_carlo_simulation.py.

For each synthetic tomato season:
  1. Bootstrap contiguous day-blocks from the Stuard feed to build a season-
     length weather series (preserves 10-min autocorrelation).
  2. Initialize VWC at field capacity.
  3. At each 10-min step, update state:
        dVWC = (rain - etc + irrigation_mm_this_step) * WETTING_VWC_PER_MM
        VWC = clip(VWC + dVWC, WP, FC)  # no deep drainage above FC
  4. Policy under test chooses irrigation at each hourly decision point using
     the forecast of its choice (persistence by default).
  5. Track per-season totals: water mm/line, stress hours below WP, cumulative
     deficit %-hours below MAD_lo.

Reports mean ± 2-sigma CI per policy across --seasons runs.

Backing: replaces the classifier-level IID Monte Carlo (which had no soil
state and no temporal structure), per plan Phase 4.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    ACTION_VOLUME_MM,
    DATA_PROCESSED,
    FIELD_CAPACITY_PCT,
    HORIZON_LABELS,
    IRRIFRAME_SEASON_MM,
    REPORTS,
    WILTING_POINT_PCT,
)
from src.policy.policy_rule import mad_threshold

DATASET = DATA_PROCESSED / "modeling_dataset_v2.parquet"

STEPS_PER_HOUR = 6
STEPS_PER_DAY = 144
SEASON_DAYS = 70
WETTING_VWC_PER_MM = 100.0 / 200.0  # 1 mm -> 0.5 VWC% for 20cm depth


@dataclass
class SeasonResult:
    policy: str
    total_mm: float
    stress_hours_below_wp: float
    deficit_pct_hours_below_mad: float
    actions: dict[str, int]


def _synthesize_season(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap contiguous day-blocks from df to build SEASON_DAYS days."""
    dates = np.array(sorted(df["datetime"].dt.floor("D").unique()))
    if len(dates) == 0:
        raise SystemExit("Empty dataset; cannot bootstrap.")
    chosen = rng.choice(dates, size=SEASON_DAYS, replace=True)
    blocks = []
    for d in chosen:
        block = df[df["datetime"].dt.floor("D") == d]
        if len(block) < STEPS_PER_DAY:
            block = block.iloc[: STEPS_PER_DAY] if len(block) > 0 else block
        blocks.append(block.head(STEPS_PER_DAY).reset_index(drop=True))
    season = pd.concat(blocks, ignore_index=True)
    season["_season_step"] = np.arange(len(season))
    return season


def _decide_fao56(vwc: float, depleted_mm: float, stage: str) -> tuple[str, float]:
    frac = {"initial": 0.30, "development": 0.40, "mid": 0.40, "late": 0.50}.get(stage, 0.40)
    taw = FIELD_CAPACITY_PCT - WILTING_POINT_PCT
    refill = frac * taw
    if depleted_mm >= refill:
        return "ON_HIGH", ACTION_VOLUME_MM["ON_HIGH"]
    if depleted_mm >= 0.5 * refill:
        return "ON_LOW", ACTION_VOLUME_MM["ON_LOW"]
    return "OFF", 0.0


def _decide_rule_persistence(vwc_hat_3h: float, stage: str) -> tuple[str, float]:
    hi = mad_threshold(stage)
    lo = hi - 2.0
    if vwc_hat_3h < lo:
        return "ON_HIGH", ACTION_VOLUME_MM["ON_HIGH"]
    if vwc_hat_3h < hi:
        return "ON_LOW", ACTION_VOLUME_MM["ON_LOW"]
    return "OFF", 0.0


def _simulate(season: pd.DataFrame, policy: str) -> SeasonResult:
    vwc = FIELD_CAPACITY_PCT
    depleted_mm = 0.0
    total_mm = 0.0
    stress_steps = 0
    deficit_pct_steps = 0.0
    actions = {"OFF": 0, "ON_LOW": 0, "ON_HIGH": 0}

    rain = season["rain_mm_h"].fillna(0.0).to_numpy() / STEPS_PER_HOUR  # mm per 10-min
    etc_h = season["etc_mm_h"].fillna(0.0).to_numpy()
    stages = season["growth_stage"].astype(str).to_numpy()

    for i in range(len(season)):
        step_etc = etc_h[i] / STEPS_PER_HOUR
        irr_mm = 0.0
        if i % STEPS_PER_HOUR == 0:  # hourly decision
            if policy == "fao56":
                action, vol = _decide_fao56(vwc, depleted_mm, stages[i])
            elif policy == "rule_persistence":
                action, vol = _decide_rule_persistence(vwc, stages[i])
            else:
                raise SystemExit(f"Unknown policy: {policy}")
            actions[action] += 1
            irr_mm = vol
            if vol > 0:
                depleted_mm = 0.0

        dvwc = (rain[i] - step_etc + irr_mm) * WETTING_VWC_PER_MM
        vwc = float(np.clip(vwc + dvwc, WILTING_POINT_PCT, FIELD_CAPACITY_PCT))
        depleted_mm = max(depleted_mm + step_etc - rain[i] - irr_mm, 0.0)
        total_mm += irr_mm

        if vwc <= WILTING_POINT_PCT + 1e-6:
            stress_steps += 1
        mad_lo = mad_threshold(stages[i]) - 2.0
        if vwc < mad_lo:
            deficit_pct_steps += (mad_lo - vwc)

    return SeasonResult(
        policy=policy,
        total_mm=total_mm,
        stress_hours_below_wp=stress_steps / STEPS_PER_HOUR,
        deficit_pct_hours_below_mad=deficit_pct_steps / STEPS_PER_HOUR,
        actions=actions,
    )


def _confidence_band(values: np.ndarray) -> tuple[float, float, float]:
    m = float(np.mean(values))
    s = float(np.std(values))
    return m, m - 2.0 * s, m + 2.0 * s


def run(seasons: int = 1000, seed: int = 0) -> dict:
    df = pd.read_parquet(DATASET)
    rng = np.random.default_rng(seed)

    policies = ["fao56", "rule_persistence"]
    by_policy: dict[str, list[SeasonResult]] = {p: [] for p in policies}

    for _ in range(seasons):
        season = _synthesize_season(df, rng)
        for p in policies:
            by_policy[p].append(_simulate(season, p))

    summary: dict[str, dict] = {}
    for p, results in by_policy.items():
        mm = np.array([r.total_mm for r in results])
        stress = np.array([r.stress_hours_below_wp for r in results])
        deficit = np.array([r.deficit_pct_hours_below_mad for r in results])
        mm_mean, mm_lo, mm_hi = _confidence_band(mm)
        st_mean, st_lo, st_hi = _confidence_band(stress)
        df_mean, df_lo, df_hi = _confidence_band(deficit)
        irri = float(IRRIFRAME_SEASON_MM)
        savings_pct = (irri - mm) / irri * 100.0
        sv_mean, sv_lo, sv_hi = _confidence_band(savings_pct)
        summary[p] = {
            "seasons": len(results),
            "total_mm_mean": mm_mean, "total_mm_ci95": [mm_lo, mm_hi],
            "stress_hours_mean": st_mean, "stress_hours_ci95": [st_lo, st_hi],
            "deficit_pct_hours_mean": df_mean, "deficit_pct_hours_ci95": [df_lo, df_hi],
            "savings_vs_irriframe_pct_mean": sv_mean,
            "savings_vs_irriframe_pct_ci95": [sv_lo, sv_hi],
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = run(seasons=args.seasons, seed=args.seed)
    (REPORTS / "monte_carlo.json").write_text(json.dumps(summary, indent=2, default=float))

    md = ["# Physics-based Monte Carlo (Phase 4)", "",
          f"Seasons simulated: **{args.seasons}** · season length {SEASON_DAYS} days · seed {args.seed}.",
          f"Weather bootstrapped from `data/processed/modeling_dataset_v2.parquet` (day-block).",
          f"IRRIFRAME reference: **{IRRIFRAME_SEASON_MM} mm** per plot, full 2023 season.",
          "",
          "| policy | mm/season (mean) | 95% CI | stress h<WP (mean) | deficit %·h<MAD (mean) | savings vs IRRIFRAME (mean %) |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for p_name, s in summary.items():
        md.append(
            f"| {p_name} | {s['total_mm_mean']:.1f} | "
            f"[{s['total_mm_ci95'][0]:.1f}, {s['total_mm_ci95'][1]:.1f}] | "
            f"{s['stress_hours_mean']:.1f} | {s['deficit_pct_hours_mean']:.1f} | "
            f"{s['savings_vs_irriframe_pct_mean']:+.1f} |"
        )
    md += ["",
           "## Honest reading",
           "- `rule_persistence` uses `vwc_hat(t+3h) = vwc(t)` as its forecast; no learned model runs inside the MC. Coupling the LSTM inside the MC is Phase 6 work.",
           "- `savings_vs_irriframe_pct` can go negative (over-watering) or close to 100% (under-watering causing crop stress). Read alongside the stress/deficit columns — saving water while starving the crop is not a win.",
           "- Unlike the legacy v1 MC, state is preserved across steps and the VWC trajectory responds to rain and ETc. Legacy numbers were IID samples; these are not comparable."]
    (REPORTS / "monte_carlo.md").write_text("\n".join(md))
    print(f"Wrote {REPORTS / 'monte_carlo.md'}")


if __name__ == "__main__":
    main()
