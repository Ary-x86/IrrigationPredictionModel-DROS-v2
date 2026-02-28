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


def generate_event_based_labels(
    df: pd.DataFrame,
    lower_limit_standard: float,
    upper_limit_standard: float,
    critical_limit: float,
) -> pd.DataFrame:
    """
    Generate control ACTION labels, not persistent state labels.

    Why this matters:
    - A 10-minute control loop should treat ON as a command pulse, not as a state
      that gets repeated every row while the soil remains dry.
    - Likewise, OFF should represent an action when entering the over-capacity zone,
      not be spammed continuously.
    - No Adjustment (2) is the default when we are already in a regime and no new action
      should be issued.

    Class semantics:
    0 = OFF       -> send one OFF action when entering the wet / over-capacity zone
    1 = ON        -> send one ON action when entering the dry zone and no rain is expected
    2 = No Adj    -> do nothing / hold current behavior
    3 = Alert     -> critically dry, but rain is expected soon
    """
    df = df.copy()
    df["Irrigation_Decision"] = 2  # default No Adjustment

    for line, group in df.groupby("line", sort=False):
        # We walk line-by-line through time because this is control logic with memory.
        idxs = list(group.sort_values("datetime").index)

        # These flags track whether we are already "inside" a dry/wet regime.
        # That prevents repeated ON/OFF spam every 10 minutes.
        dry_active = False
        wet_active = False

        for idx in idxs:
            moisture = float(df.at[idx, "Soil Moisture [RH%]"])
            rain = float(df.at[idx, "Weather Forecast Rainfall [mm]"])

            is_dry = moisture < lower_limit_standard
            is_wet = moisture > upper_limit_standard
            is_critical = moisture < critical_limit
            rain_expected = rain > 2.0

            decision = 2  # default No Adjustment

            # ALERT has highest priority:
            # If soil is critically low but relevant rain is expected,
            # the system warns instead of blindly firing repeated irrigation commands.
            if is_dry and rain_expected and is_critical:
                decision = 3

                # We remain logically in the dry regime.
                dry_active = True
                wet_active = False

            # Dry zone and no meaningful rain expected:
            # Trigger ONE ON action when entering the dry regime.
            elif is_dry and not rain_expected:
                if not dry_active:
                    decision = 1
                    dry_active = True
                    wet_active = False
                else:
                    # Already in dry regime -> do not repeat ON every row.
                    decision = 2

            # Wet / over-capacity zone:
            # Trigger ONE OFF action when entering the wet regime.
            elif is_wet:
                if not wet_active:
                    decision = 0
                    wet_active = True
                    dry_active = False
                else:
                    # Already in wet regime -> do not repeat OFF every row.
                    decision = 2

            else:
                # Back in the neutral middle band:
                # no special state is currently active.
                dry_active = False
                wet_active = False
                decision = 2

            df.at[idx, "Irrigation_Decision"] = decision

    return df


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
    # This is our practical approximation of the slow drainage segment used to estimate
    # the soil capacity point.
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

    print("Generating AI target classes using EVENT-BASED control logic...")
    df = generate_event_based_labels(
        df=df,
        lower_limit_standard=lower_limit_standard,
        upper_limit_standard=upper_limit_standard,
        critical_limit=critical_limit,
    )

    class_counts = (
        df["Irrigation_Decision"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("Class counts:")
    print(class_counts)

    total_rows = len(df)
    if total_rows > 0:
        on_ratio = class_counts.get(1, 0) / total_rows
        print(f"ON class ratio: {on_ratio:.4%}")

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
        "labeling_mode": "event_based_control_actions",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved full labeled dataset to {modeling_path}")
    print(f"Saved training dataset to {processed_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    calculate_capacity_and_merge()