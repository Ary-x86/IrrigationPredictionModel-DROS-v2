# src/03_soil_capacity_calculator.py

from pathlib import Path
import json
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# The paper describes filtering the slow-decline region after irrigation or rain
# to estimate soil capacity, but it does not publish a hard numeric cutoff.
# So we make that heuristic explicit and tunable here.
RECESSION_LOOKBACK_STEPS = 12   # 12 x 10 min = 2 hours
MAX_SLOW_DRAINAGE_DROP = -0.5   # keep only mild negative slopes close to 0


def calculate_capacity_and_merge():
    print("Loading intermediate datasets...")

    df_sensors = pd.read_csv(DATA_DIR / "merged_sensor_data.csv")
    df_weather = pd.read_csv(DATA_DIR / "open_meteo_forecast_data.csv")

    df_sensors["datetime"] = pd.to_datetime(df_sensors["datetime"])
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])

    print("Merging hourly weather into 10-minute sensor data...")
    df_sensors["join_hour"] = df_sensors["datetime"].dt.floor("h")
    df_weather["join_hour"] = df_weather["datetime"]

    df = (
        df_sensors
        .merge(df_weather, on="join_hour", how="left", suffixes=("", "_weather"))
        .drop(columns=["join_hour", "datetime_weather"])
        .sort_values(["line", "datetime"])
        .reset_index(drop=True)
    )

    weather_required = [
        "Weather Forecast Rainfall [mm]",
        "Crop Data Evapotranspiration [mm]",
        "Weather Forecast Environmental humidity [RH %]",
    ]
    df = df.dropna(subset=weather_required).copy()

    print("Calculating soil moisture derivative per irrigation line...")
    df["Moisture_Derivative"] = df.groupby("line")["Soil Moisture [RH%]"].diff()

    # Only consider the moisture recession region shortly after irrigation/rain events.
    recent_irrigation = (
        df.groupby("line")["Irrigation (ON/OFF)"]
        .transform(lambda s: s.rolling(RECESSION_LOOKBACK_STEPS, min_periods=1).max())
        .gt(0)
    )
    recent_rain = (
        df.groupby("line")["Weather Forecast Rainfall [mm]"]
        .transform(lambda s: s.rolling(RECESSION_LOOKBACK_STEPS, min_periods=1).max())
        .gt(0)
    )

    slow_drainage_mask = (
        (recent_irrigation | recent_rain)
        & df["Moisture_Derivative"].lt(0)
        & df["Moisture_Derivative"].ge(MAX_SLOW_DRAINAGE_DROP)
    )

    capacity_data = df.loc[slow_drainage_mask, "Soil Moisture [RH%]"].dropna()

    # Fallback in case the strict filter is too aggressive on a future dataset.
    if capacity_data.empty:
        print("Warning: strict slow-drainage filter returned no rows; falling back to basic derivative filter.")
        fallback_mask = (
            df["Moisture_Derivative"].lt(0)
            & df["Moisture_Derivative"].ge(MAX_SLOW_DRAINAGE_DROP)
        )
        capacity_data = df.loc[fallback_mask, "Soil Moisture [RH%]"].dropna()

    mu = float(capacity_data.mean())
    sigma = float(capacity_data.std())

    lower_limit_standard = mu - (sigma / 2.0)
    upper_limit_standard = mu + (sigma / 2.0)
    critical_limit = mu - sigma

    print(f"Calculated Soil Capacity (mu): {mu:.4f}%")
    print(f"Calculated Standard Deviation (sigma): {sigma:.4f}%")
    print(f"Lower standard limit (mu - sigma/2): {lower_limit_standard:.4f}%")
    print(f"Upper standard limit (mu + sigma/2): {upper_limit_standard:.4f}%")
    print(f"Critical limit (mu - sigma): {critical_limit:.4f}%")

    print("Generating AI target classes using the paper's flow logic...")

    # Default = No Adjustment (2)
    df["Irrigation_Decision"] = 2

    dry_mask = df["Soil Moisture [RH%]"] < lower_limit_standard
    wet_mask = df["Soil Moisture [RH%]"] > upper_limit_standard
    rain_mask = df["Weather Forecast Rainfall [mm]"] > 2.0
    critical_dry_mask = df["Soil Moisture [RH%]"] < critical_limit

    # Condition 1: Above the upper limit -> OFF
    df.loc[wet_mask, "Irrigation_Decision"] = 0

    # Condition 2: Below the lower limit and no significant rain -> ON
    df.loc[dry_mask & (~rain_mask), "Irrigation_Decision"] = 1

    # Condition 3: Below the lower limit, rain is coming, and critically low -> ALERT
    df.loc[dry_mask & rain_mask & critical_dry_mask, "Irrigation_Decision"] = 3

    class_counts = (
        df["Irrigation_Decision"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("Class counts:")
    print(class_counts)

    # Full labeled dataset for debugging, diagnostics, and Monte Carlo simulation.
    full_columns = [
        "datetime",
        "line",
        "Irrigation (ON/OFF)",
        "Soil Moisture [RH%]",
        "Soil Temperature [C]",
        "Environmental Temperature [ C]",
        "Environmental Humidity [RH %]",
        "Weather Forecast Rainfall [mm]",
        "Crop Data Evapotranspiration [mm]",
        "Irrigation_Decision",
        "Moisture_Derivative",
    ]
    df_full = df[full_columns].dropna().copy()

    # Final training dataset: keep only the six paper-style features + target.
    feature_columns = [
        "Soil Moisture [RH%]",
        "Soil Temperature [C]",
        "Environmental Temperature [ C]",
        "Environmental Humidity [RH %]",
        "Weather Forecast Rainfall [mm]",
        "Crop Data Evapotranspiration [mm]",
        "Irrigation_Decision",
    ]
    df_train = df_full[feature_columns].copy()

    modeling_path = DATA_DIR / "modeling_dataset_full.csv"
    processed_path = DATA_DIR / "processed_dataset.csv"
    metadata_path = DATA_DIR / "soil_capacity_metadata.json"

    df_full.to_csv(modeling_path, index=False)
    df_train.to_csv(processed_path, index=False)

    metadata = {
        "mu": mu,
        "sigma": sigma,
        "lower_limit_standard": lower_limit_standard,
        "upper_limit_standard": upper_limit_standard,
        "critical_limit": critical_limit,
        "recession_lookback_steps": RECESSION_LOOKBACK_STEPS,
        "max_slow_drainage_drop": MAX_SLOW_DRAINAGE_DROP,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved full labeled dataset to {modeling_path}")
    print(f"Saved training dataset to {processed_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    calculate_capacity_and_merge()