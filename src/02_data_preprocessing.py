# src/02_data_preprocessing.py

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PAPER_START = pd.Timestamp("2023-07-28 00:00:00")
PAPER_END = pd.Timestamp("2023-09-03 23:59:59")


def read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    # Some exported CSVs contain accidental repeated header rows inside the file.
    if "ts_generation" in df.columns:
        df = df[df["ts_generation"] != "ts_generation"].copy()

    return df


def coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_generation"] = pd.to_numeric(df["ts_generation"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts_generation"], unit="ms", errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    # Restrict to the paper's reference window so sensor data and weather overlap correctly.
    df = df[(df["datetime"] >= PAPER_START) & (df["datetime"] <= PAPER_END)].copy()
    return df


def prepare_environmental(df_env: pd.DataFrame) -> pd.DataFrame:
    df_env = coerce_timestamp(df_env)

    df_env["temperature"] = pd.to_numeric(df_env["temperature"], errors="coerce")
    df_env["humidity"] = pd.to_numeric(df_env["humidity"], errors="coerce")

    df_env["datetime"] = df_env["datetime"].dt.round("10min")

    # Environmental data is shared across all irrigation lines.
    df_env = (
        df_env.groupby("datetime", as_index=False)
        .agg(
            {
                "temperature": "mean",
                "humidity": "mean",
            }
        )
        .rename(
            columns={
                "temperature": "Environmental Temperature [ C]",
                "humidity": "Environmental Humidity [RH %]",
            }
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df_env


def prepare_soil(df_soil: pd.DataFrame) -> pd.DataFrame:
    df_soil = coerce_timestamp(df_soil)

    df_soil["line"] = df_soil["line"].astype(str)
    df_soil["humidity"] = pd.to_numeric(df_soil["humidity"], errors="coerce")
    df_soil["temperature"] = pd.to_numeric(df_soil["temperature"], errors="coerce")
    df_soil["electrical_conductivity"] = pd.to_numeric(df_soil["electrical_conductivity"], errors="coerce")

    df_soil["datetime"] = df_soil["datetime"].dt.round("10min")

    df_soil = (
        df_soil.groupby(["line", "datetime"], as_index=False)
        .agg(
            {
                "humidity": "mean",
                "temperature": "mean",
                "electrical_conductivity": "mean",
            }
        )
        .rename(
            columns={
                "humidity": "Soil Moisture [RH%]",
                "temperature": "Soil Temperature [C]",
                "electrical_conductivity": "Soil Electrical Conductivity",
            }
        )
        .sort_values(["line", "datetime"])
        .reset_index(drop=True)
    )

    return df_soil


def prepare_water(df_water: pd.DataFrame) -> pd.DataFrame:
    df_water = coerce_timestamp(df_water)

    df_water["line"] = df_water["line"].astype(str)
    df_water["current_volume"] = pd.to_numeric(df_water["current_volume"], errors="coerce")

    df_water = df_water.sort_values(["line", "datetime"]).copy()
    df_water["datetime"] = df_water["datetime"].dt.round("10min")

    # We keep the last cumulative counter reading per 10-minute bin.
    df_water = (
        df_water.groupby(["line", "datetime"], as_index=False)
        .agg({"current_volume": "last"})
        .sort_values(["line", "datetime"])
        .reset_index(drop=True)
    )

    return df_water


def build_line_grids(df_env: pd.DataFrame, df_soil: pd.DataFrame, df_water: pd.DataFrame) -> pd.DataFrame:
    env_start = df_env["datetime"].min().ceil("10min")
    env_end = df_env["datetime"].max().floor("10min")

    lines = sorted(set(df_soil["line"]).intersection(set(df_water["line"])))
    grids = []

    for line in lines:
        soil_line = df_soil[df_soil["line"] == line]
        water_line = df_water[df_water["line"] == line]

        line_start = max(
            env_start,
            soil_line["datetime"].min().ceil("10min"),
            water_line["datetime"].min().ceil("10min"),
        )
        line_end = min(
            env_end,
            soil_line["datetime"].max().floor("10min"),
            water_line["datetime"].max().floor("10min"),
        )

        idx = pd.date_range(line_start, line_end, freq="10min")
        grid = pd.DataFrame({"datetime": idx})
        grid["line"] = line
        grids.append(grid)

    return pd.concat(grids, ignore_index=True)


def load_and_clean_data():
    print("Loading raw CSV files...")

    df_env_raw = read_raw_csv(DATA_DIR / "stuard_environmental_data.csv")
    df_soil_raw = read_raw_csv(DATA_DIR / "stuard_soil_data.csv")
    df_water_raw = read_raw_csv(DATA_DIR / "stuard_water_meter_data.csv")

    print("Processing Environmental Data...")
    df_env = prepare_environmental(df_env_raw)

    print("Processing Soil Data...")
    df_soil = prepare_soil(df_soil_raw)

    print("Processing Water Meter Data...")
    df_water = prepare_water(df_water_raw)

    print("Building a proper 10-minute grid for ALL irrigation lines...")
    df_grid = build_line_grids(df_env, df_soil, df_water)

    print("Merging datasets...")
    df_merged = (
        df_grid
        .merge(df_soil, on=["line", "datetime"], how="left")
        .merge(df_water, on=["line", "datetime"], how="left")
        .merge(df_env, on="datetime", how="left")
        .sort_values(["line", "datetime"])
        .reset_index(drop=True)
    )

    # Forward-fill line-dependent variables within each irrigation line only.
    line_cols = [
        "Soil Moisture [RH%]",
        "Soil Temperature [C]",
        "Soil Electrical Conductivity",
        "current_volume",
    ]
    for col in line_cols:
        df_merged[col] = df_merged.groupby("line")[col].ffill()

    # Environmental data is shared, so a simple forward-fill is fine.
    env_cols = [
        "Environmental Temperature [ C]",
        "Environmental Humidity [RH %]",
    ]
    for col in env_cols:
        df_merged[col] = df_merged[col].ffill()

    df_merged = df_merged.dropna(
        subset=[
            "Soil Moisture [RH%]",
            "Soil Temperature [C]",
            "Soil Electrical Conductivity",
            "current_volume",
            "Environmental Temperature [ C]",
            "Environmental Humidity [RH %]",
        ]
    ).copy()

    # Water meter is cumulative, so irrigation activity is detected from positive jumps.
    df_merged["volume_diff"] = (
        df_merged.groupby("line")["current_volume"].diff().fillna(0.0)
    )
    df_merged["Irrigation (ON/OFF)"] = (df_merged["volume_diff"] > 0).astype(int)

    df_merged["Daily Hour"] = df_merged["datetime"].dt.hour

    final_columns = [
        "datetime",
        "line",
        "Irrigation (ON/OFF)",
        "current_volume",
        "volume_diff",
        "Soil Moisture [RH%]",
        "Soil Temperature [C]",
        "Soil Electrical Conductivity",
        "Daily Hour",
        "Environmental Temperature [ C]",
        "Environmental Humidity [RH %]",
    ]
    df_merged = df_merged[final_columns].copy()

    print(f"Data processing complete! Final dataset shape: {df_merged.shape}")
    print("\nRows per line:")
    print(df_merged["line"].value_counts().sort_index())

    output_path = DATA_DIR / "merged_sensor_data.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"Saved merged sensor data to {output_path}")


if __name__ == "__main__":
    load_and_clean_data()