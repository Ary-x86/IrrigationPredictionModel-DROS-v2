"""Assemble the Track B modeling dataset.

Reads Stuard sensors + Open-Meteo hourly forecast, produces the canonical
feature table with targets, writes data/processed/modeling_dataset_v2.parquet.

This is the single source of truth for Track B — all downstream training,
evaluation, and policy code reads this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, LAT
from src.data_io.load_stuard import load_and_join
from src.features.et import etc_mm_per_hour, hargreaves_et0_mm_day, vpd_kpa
from src.features.phenology import gdd_cumulative, kc_from_gdd, stage_from_gdd
from src.features.radiation import daylength_hours, extraterrestrial_radiation
from src.features.soil import add_vwc_features
from src.features.time_features import add_time_features
from src.features.weather import add_rain_rolls, hours_since_last_irrigation
from src.labels.derive_targets import add_future_vwc, target_columns

RAW_COL_RENAMES = {
    "Soil Moisture [RH%]": "vwc_20cm",
    "Soil Temperature [C]": "soil_temp_c",
    "Environmental Temperature [ C]": "air_temp_c",
    "Environmental Humidity [RH %]": "rh_pct",
    "Weather Forecast Rainfall [mm]": "rain_mm_h",
    "Crop Data Evapotranspiration [mm]": "et0_open_meteo_mm_h",
}

FEATURE_COLUMNS = [
    "vwc_20cm",
    "vwc_20cm_lag_6s", "vwc_20cm_lag_18s", "vwc_20cm_lag_36s",
    "vwc_20cm_lag_72s", "vwc_20cm_lag_144s",
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
META_COLUMNS = ["datetime", "line", "growth_stage", "volume_diff"]


def build() -> pd.DataFrame:
    df = load_and_join()
    # fill-in basics
    df["Weather Forecast Rainfall [mm]"] = df["Weather Forecast Rainfall [mm]"].fillna(0.0)
    df["Crop Data Evapotranspiration [mm]"] = df["Crop Data Evapotranspiration [mm]"].fillna(0.0)
    if "Weather Forecast Environmental humidity [RH %]" in df.columns:
        df["Weather Forecast Environmental humidity [RH %]"] = (
            df["Weather Forecast Environmental humidity [RH %]"]
            .fillna(df["Environmental Humidity [RH %]"])
        )

    # VPD needs raw T + RH names before rename
    df["vpd_kpa"] = vpd_kpa(df["Environmental Temperature [ C]"], df["Environmental Humidity [RH %]"])

    # astronomy
    doy = pd.to_datetime(df["datetime"]).dt.dayofyear.to_numpy()
    df["ra_mj_m2_day"] = extraterrestrial_radiation(doy, LAT)
    df["daylength_h"] = daylength_hours(doy, LAT)

    # daily temp stats for Hargreaves
    df["_date"] = pd.to_datetime(df["datetime"]).dt.date
    grp_t = df.groupby("_date")["Environmental Temperature [ C]"]
    df["et0_hargreaves_mm_day"] = hargreaves_et0_mm_day(
        grp_t.transform("mean"),
        grp_t.transform("max"),
        grp_t.transform("min"),
        df["ra_mj_m2_day"].to_numpy(),
    )

    # phenology
    df["gdd_cum"] = gdd_cumulative(df["datetime"], df["Environmental Temperature [ C]"]).astype(float)
    df["growth_stage"] = df["gdd_cum"].apply(stage_from_gdd)
    df["kc_dynamic"] = df["gdd_cum"].apply(kc_from_gdd)
    df["etc_mm_h"] = etc_mm_per_hour(df["kc_dynamic"], df["Crop Data Evapotranspiration [mm]"])

    # time encodings
    df = add_time_features(df)

    # rename canonical columns
    df = df.rename(columns=RAW_COL_RENAMES)

    # soil lags/rolls, rain rolls, irrigation clock
    df = add_vwc_features(df, group_col="line", vwc_col="vwc_20cm")
    df = add_rain_rolls(df, group_col="line", rain_col="rain_mm_h")
    df["volume_diff"] = df["volume_diff"].fillna(0.0)
    df["hours_since_last_irrigation"] = hours_since_last_irrigation(
        df, group_col="line", time_col="datetime", irr_col="volume_diff",
    )
    df["line_id"] = df["line"] - df["line"].min()

    # targets
    df = add_future_vwc(df, group_col="line", vwc_col="vwc_20cm")

    # drop rows missing any lag feature (start-of-line warm-up)
    df = df.dropna(subset=FEATURE_COLUMNS).copy()

    out = df[META_COLUMNS + FEATURE_COLUMNS + target_columns()].reset_index(drop=True)
    return out


def save(df: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / "modeling_dataset_v2.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows x {len(df.columns)} cols to {path}")


if __name__ == "__main__":
    df = build()
    save(df)
    print(df[target_columns()].describe())
