"""Feature engineering for Track A (GBDT).

Consumes the already-cleaned per-line 10-minute grid from data/merged_sensor_data.csv
and the hourly Open-Meteo archive, emits a flat feature table with strictly
past-only features plus future-VWC targets for horizons {1,3,6,12,24}h.

Target is future soil moisture, not a threshold-derived class, so the original
label-leakage route is structurally closed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.gbdt.config import (
    EXP_DATA,
    GDD_STAGE_EDGES,
    HORIZONS_10MIN,
    HORIZON_LABELS,
    KC_BY_STAGE,
    LAT,
    LOOKBACK_STEPS,
    PROJECT_DATA,
    T_BASE_C,
    TRANSPLANT_DATE,
)

SOIL_COL = "Soil Moisture [RH%]"
SOIL_T_COL = "Soil Temperature [C]"
AIR_T_COL = "Environmental Temperature [ C]"
RH_COL = "Environmental Humidity [RH %]"
RAIN_COL = "Weather Forecast Rainfall [mm]"
ET0_COL = "Crop Data Evapotranspiration [mm]"
RH_FCAST_COL = "Weather Forecast Environmental humidity [RH %]"


def _vpd_kpa(t_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    ea = es * rh_pct / 100.0
    return (es - ea).clip(lower=0.0)


def _extraterrestrial_radiation(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    gsc = 0.0820
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    cos_ws = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    ws = np.arccos(cos_ws)
    ra = (24.0 * 60.0 / np.pi) * gsc * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )
    return ra


def _daylength_hours(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    cos_ws = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    ws = np.arccos(cos_ws)
    return (24.0 / np.pi) * ws


def _hargreaves_et0(t_mean: pd.Series, t_max: pd.Series, t_min: pd.Series, ra_mj: np.ndarray) -> pd.Series:
    delta_t = (t_max - t_min).clip(lower=0.0)
    return 0.0023 * (t_mean + 17.8) * np.sqrt(delta_t) * ra_mj * 0.408


def _gdd_cumulative(timestamps: pd.Series, t_mean_daily: pd.Series, t_base: float) -> pd.Series:
    daily = t_mean_daily.groupby(timestamps.dt.date).transform("mean")
    daily_gdd = (daily - t_base).clip(lower=0.0)
    cum = (
        daily_gdd.groupby(timestamps.dt.date)
        .transform("first")
        .groupby(timestamps.dt.date)
        .transform("first")
    )
    per_day = (
        pd.DataFrame({"date": timestamps.dt.date, "v": daily_gdd})
        .drop_duplicates("date")
        .sort_values("date")
    )
    per_day["cum"] = per_day["v"].cumsum()
    mapping = dict(zip(per_day["date"], per_day["cum"]))
    return timestamps.dt.date.map(mapping).astype(float)


def _stage_from_gdd(gdd: float) -> str:
    for lo, hi, name in GDD_STAGE_EDGES:
        if lo <= gdd < hi:
            return name
    return GDD_STAGE_EDGES[-1][2]


def _kc_from_gdd(gdd: float) -> float:
    stages = GDD_STAGE_EDGES
    centers = [(lo + hi) / 2.0 for lo, hi, _ in stages]
    kcs = [KC_BY_STAGE[name] for _, _, name in stages]
    if gdd <= centers[0]:
        return kcs[0]
    if gdd >= centers[-1]:
        return kcs[-1]
    return float(np.interp(gdd, centers, kcs))


def _hours_since_last_irrigation(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("line", sort=False):
        ts = g["datetime"].to_numpy()
        irr = g["volume_diff"].fillna(0.0).to_numpy() > 0.0
        last_ts = None
        hours = np.full(len(g), np.nan, dtype=float)
        for i in range(len(g)):
            if irr[i]:
                last_ts = ts[i]
            if last_ts is not None:
                hours[i] = (ts[i] - last_ts) / np.timedelta64(1, "h")
        out.loc[g.index] = hours
    return out.fillna(out.median() if out.notna().any() else 0.0)


def build_features() -> pd.DataFrame:
    sensors = pd.read_csv(PROJECT_DATA / "merged_sensor_data.csv")
    weather = pd.read_csv(PROJECT_DATA / "open_meteo_forecast_data.csv")
    sensors["datetime"] = pd.to_datetime(sensors["datetime"])
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    sensors["line"] = sensors["line"].astype(int)

    sensors["join_hour"] = sensors["datetime"].dt.floor("h")
    weather = weather.rename(columns={"datetime": "join_hour"})
    df = sensors.merge(weather, on="join_hour", how="left").drop(columns=["join_hour"])
    df = df.sort_values(["line", "datetime"]).reset_index(drop=True)

    df[RAIN_COL] = df[RAIN_COL].fillna(0.0)
    df[ET0_COL] = df[ET0_COL].fillna(0.0)
    df[RH_FCAST_COL] = df[RH_FCAST_COL].fillna(df[RH_COL])

    df["vpd_kpa"] = _vpd_kpa(df[AIR_T_COL], df[RH_COL])

    doy = df["datetime"].dt.dayofyear.to_numpy()
    df["ra_mj_m2_day"] = _extraterrestrial_radiation(doy, LAT)
    df["daylength_h"] = _daylength_hours(doy, LAT)

    df["date"] = df["datetime"].dt.date
    daily_tmax = df.groupby("date")[AIR_T_COL].transform("max")
    daily_tmin = df.groupby("date")[AIR_T_COL].transform("min")
    daily_tmean = df.groupby("date")[AIR_T_COL].transform("mean")
    df["et0_hargreaves_mm_day"] = _hargreaves_et0(daily_tmean, daily_tmax, daily_tmin, df["ra_mj_m2_day"].to_numpy())

    transplant = pd.Timestamp(TRANSPLANT_DATE).date()
    per_day_mean = df.groupby("date")[AIR_T_COL].mean()
    per_day_gdd = (per_day_mean - T_BASE_C).clip(lower=0.0)
    per_day_gdd = per_day_gdd.sort_index()
    per_day_gdd = per_day_gdd[per_day_gdd.index >= transplant]
    per_day_gdd_cum = per_day_gdd.cumsum()
    gdd_map = per_day_gdd_cum.to_dict()
    df["gdd_cum"] = df["date"].map(gdd_map).astype(float)

    df["growth_stage"] = df["gdd_cum"].apply(_stage_from_gdd)
    df["kc_dynamic"] = df["gdd_cum"].apply(_kc_from_gdd)
    df["etc_mm_h"] = df["kc_dynamic"] * df[ET0_COL]

    df["doy"] = doy
    df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    df["sin_doy"] = np.sin(2.0 * np.pi * df["doy"] / 365.0)
    df["cos_doy"] = np.cos(2.0 * np.pi * df["doy"] / 365.0)
    df["sin_hour"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)

    df = df.rename(columns={
        SOIL_COL: "vwc_20cm",
        SOIL_T_COL: "soil_temp_c",
        AIR_T_COL: "air_temp_c",
        RH_COL: "rh_pct",
        RAIN_COL: "rain_mm_h",
        ET0_COL: "et0_open_meteo_mm_h",
    })

    g = df.groupby("line", sort=False)
    for h in [6, 18, 36, 72, 144]:
        df[f"vwc_20cm_lag_{h}s"] = g["vwc_20cm"].shift(h)
    df["vwc_20cm_roll_mean_3h"] = g["vwc_20cm"].transform(lambda s: s.rolling(18, min_periods=1).mean())
    df["vwc_20cm_roll_std_6h"] = g["vwc_20cm"].transform(lambda s: s.rolling(36, min_periods=2).std()).fillna(0.0)
    df["vwc_20cm_deriv_1h"] = g["vwc_20cm"].transform(lambda s: s.diff(6)).fillna(0.0)

    df["rain_mm_1h_sum"] = g["rain_mm_h"].transform(lambda s: s.rolling(6, min_periods=1).sum())
    df["rain_mm_6h_sum"] = g["rain_mm_h"].transform(lambda s: s.rolling(36, min_periods=1).sum())
    df["rain_mm_24h_sum"] = g["rain_mm_h"].transform(lambda s: s.rolling(144, min_periods=1).sum())

    df["volume_diff"] = df["volume_diff"].fillna(0.0)
    df["hours_since_last_irrigation"] = _hours_since_last_irrigation(df)

    df["line_id"] = df["line"] - df["line"].min()

    for steps, label in zip(HORIZONS_10MIN, HORIZON_LABELS):
        df[f"y_vwc_h{label}"] = g["vwc_20cm"].shift(-steps)

    feature_cols = [
        "vwc_20cm",
        "vwc_20cm_lag_6s", "vwc_20cm_lag_18s", "vwc_20cm_lag_36s", "vwc_20cm_lag_72s", "vwc_20cm_lag_144s",
        "vwc_20cm_roll_mean_3h", "vwc_20cm_roll_std_6h", "vwc_20cm_deriv_1h",
        "soil_temp_c", "air_temp_c", "rh_pct", "vpd_kpa",
        "rain_mm_h", "rain_mm_1h_sum", "rain_mm_6h_sum", "rain_mm_24h_sum",
        "et0_open_meteo_mm_h", "et0_hargreaves_mm_day", "etc_mm_h",
        "ra_mj_m2_day", "daylength_h",
        "gdd_cum", "kc_dynamic",
        "sin_doy", "cos_doy", "sin_hour", "cos_hour",
        "hours_since_last_irrigation",
        "line_id",
    ]
    target_cols = [f"y_vwc_h{label}" for label in HORIZON_LABELS]
    meta_cols = ["datetime", "line", "growth_stage", "volume_diff"]

    df = df.dropna(subset=feature_cols).copy()

    out_cols = meta_cols + feature_cols + target_cols
    out = df[out_cols].reset_index(drop=True)
    return out


def save_features(df: pd.DataFrame) -> None:
    EXP_DATA.mkdir(parents=True, exist_ok=True)
    out_path = EXP_DATA / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows x {len(df.columns)} cols to {out_path}")


if __name__ == "__main__":
    features = build_features()
    save_features(features)
    print(features.head())
    print(features[[c for c in features.columns if c.startswith("y_vwc_h")]].describe())
