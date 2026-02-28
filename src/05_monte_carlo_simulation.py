# src/05_monte_carlo_simulation.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

    # =========================================================================
    # THE THEORY OF MONTE CARLO
    # A Monte Carlo simulation is a mathematical technique used to estimate the 
    # possible outcomes of an uncertain event. Instead of calculating one "average" 
    # season, it runs hundreds or thousands of randomized simulations using a 
    # probability distribution[cite: 335]. 
    # By running the simulation 1,000 times, the law of large numbers takes over, 
    # giving us a beautiful bell curve of all statistically probable outcomes.
    # =========================================================================


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Paper-aligned season setup.
SEASONS_TO_SIMULATE = 1000   # start with 100 while debugging if you want faster runs
DAYS_PER_SEASON = 70
SAMPLES_PER_DAY = 24 * 6     # 10-minute control loop
TOTAL_SAMPLES = DAYS_PER_SEASON * SAMPLES_PER_DAY

# Irrigation hardware.
FLOW_RATE_LPH = 300.0
LITERS_PER_10MIN = FLOW_RATE_LPH * (10.0 / 60.0)   # 50 L
PUMP_POWER_KW = 4.0
KWH_PER_10MIN = PUMP_POWER_KW * (10.0 / 60.0)

# IRRIFRAME baseline from the paper.
IRRIFRAME_MM = 324.5
AREA_M2 = 132.0
IRRIFRAME_WATER_LITERS = IRRIFRAME_MM * AREA_M2
IRRIFRAME_ENERGY_KWH = (IRRIFRAME_WATER_LITERS / FLOW_RATE_LPH) * PUMP_POWER_KW

# Prevent pathological ON-ON-ON spam when one irrigation pulse was just triggered.
COOLDOWN_STEPS = 1


def load_model_bundle():
    model_path = MODELS_DIR / "mlp_irrigation_model.pkl"
    bundle = joblib.load(model_path)
    return bundle


def load_simulation_data():
    df = pd.read_csv(DATA_DIR / "modeling_dataset_full.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def build_day_library(df: pd.DataFrame) -> list[np.ndarray]:
    """
    Build a library of daily exogenous-condition blocks.
    Each block has 144 rows x 5 columns:
        soil_temp, env_temp, env_humidity, rainfall, evapotranspiration

    We deliberately keep moisture OUT of the templates because moisture is stateful
    and must be simulated, not copied from history.
    """
    df = df.copy()
    df["date"] = df["datetime"].dt.date

    exogenous_cols = [
        "Soil Temperature [C]",
        "Environmental Temperature [ C]",
        "Environmental Humidity [RH %]",
        "Weather Forecast Rainfall [mm]",
        "Crop Data Evapotranspiration [mm]",
    ]

    day_library = []
    for _, day_df in df.groupby("date"):
        if len(day_df) == 0:
            continue

        # We want exactly 144 exogenous rows to represent one 10-minute day.
        # If a day contains more rows (e.g. multiple lines), sample 144 without replacement.
        # If it contains fewer, sample with replacement.
        sampled = day_df.sample(
            n=SAMPLES_PER_DAY,
            replace=(len(day_df) < SAMPLES_PER_DAY),
            random_state=42,
        )[exogenous_cols].to_numpy(dtype=float)

        day_library.append(sampled)

    if not day_library:
        raise ValueError("No daily exogenous blocks could be built for simulation.")

    return day_library


def estimate_transition_pools(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build empirical one-step moisture-change pools from the historical dataset.

    We estimate:
    - on_pool: moisture changes observed when irrigation should be ON
    - off_dry_pool: valve closed + no rain
    - off_rain_pool: valve closed + rain present

    This gives the Monte Carlo a state transition model instead of fake iid threshold counting.
    """
    df = df.sort_values(["line", "datetime"]).copy()

    df["next_moisture"] = df.groupby("line")["Soil Moisture [RH%]"].shift(-1)
    df["delta_to_next"] = df["next_moisture"] - df["Soil Moisture [RH%]"]
    df = df.dropna(subset=["delta_to_next"]).copy()

    on_pool = df.loc[df["Irrigation_Decision"] == 1, "delta_to_next"].to_numpy(dtype=float)

    # Sensor noise can produce tiny negative deltas even when ON.
    # Irrigation should not decrease moisture, so clip negative values away.
    on_pool = np.clip(on_pool, a_min=0.0, a_max=None)

    off_dry_pool = df.loc[
        (df["Irrigation_Decision"] != 1) & (df["Weather Forecast Rainfall [mm]"] <= 0),
        "delta_to_next"
    ].to_numpy(dtype=float)

    off_rain_pool = df.loc[
        (df["Irrigation_Decision"] != 1) & (df["Weather Forecast Rainfall [mm]"] > 0),
        "delta_to_next"
    ].to_numpy(dtype=float)

    # Safe fallbacks if any pool is empty on another dataset.
    if len(on_pool) == 0:
        on_pool = np.array([0.05], dtype=float)
    if len(off_dry_pool) == 0:
        off_dry_pool = np.array([-0.05], dtype=float)
    if len(off_rain_pool) == 0:
        off_rain_pool = off_dry_pool.copy()

    return on_pool, off_dry_pool, off_rain_pool


def bootstrap_exogenous_season(day_library: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    Build a 70-day season by sampling daily exogenous blocks with replacement.
    This preserves realistic intra-day weather/temperature structure better than iid row sampling.
    """
    season_blocks = []
    for _ in range(DAYS_PER_SEASON):
        block = day_library[rng.integers(0, len(day_library))]
        season_blocks.append(block)
    return np.vstack(season_blocks)


def make_fast_predictor(model_bundle):
    """
    Use the saved full pipeline and preserve feature names,
    so sklearn does not complain about unnamed arrays.
    """
    pipeline = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    def predict_one(feature_row: np.ndarray) -> int:
        x_df = pd.DataFrame([feature_row], columns=feature_columns)
        return int(pipeline.predict(x_df)[0])

    return predict_one


def simulate_one_season(
    predict_one,
    exogenous_season: np.ndarray,
    on_pool: np.ndarray,
    off_dry_pool: np.ndarray,
    off_rain_pool: np.ndarray,
    moisture_min: float,
    moisture_max: float,
    moisture_start: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Stateful 10-minute simulation:
    - moisture carries over through time
    - the controller sees the current simulated moisture + sampled exogenous conditions
    - ON decisions consume water/energy
    - moisture changes according to empirical transition pools
    """
    current_moisture = float(moisture_start)
    total_water_liters = 0.0
    total_energy_kwh = 0.0
    cooldown = 0

    for row in exogenous_season:
        soil_temp, env_temp, env_humidity, rain_mm, et_mm = row

        features = np.array(
            [
                current_moisture,
                soil_temp,
                env_temp,
                env_humidity,
                rain_mm,
                et_mm,
            ],
            dtype=float,
        )

        decision = predict_one(features)

        # Optional one-step cooldown to stop unrealistic spamming.
        if cooldown > 0 and decision == 1:
            decision = 2

        if decision == 1:
            total_water_liters += LITERS_PER_10MIN
            total_energy_kwh += KWH_PER_10MIN

            delta = float(rng.choice(on_pool))

            # Ensure that an ON pulse actually raises moisture at least a little.
            delta = max(delta, 0.05)
            cooldown = COOLDOWN_STEPS
        else:
            if rain_mm > 0:
                delta = float(rng.choice(off_rain_pool))
            else:
                delta = float(rng.choice(off_dry_pool))
            cooldown = max(0, cooldown - 1)

        current_moisture = float(np.clip(current_moisture + delta, moisture_min, moisture_max))

    return total_water_liters, total_energy_kwh


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean - (2.0 * std), mean + (2.0 * std)


def run_monte_carlo_simulation():
    print("--- Initiating Monte Carlo Simulation ---")
    print(f"Baseline IRRIFRAME Water Usage: {IRRIFRAME_WATER_LITERS:,.2f} Liters")
    print(f"Baseline IRRIFRAME Energy Usage: {IRRIFRAME_ENERGY_KWH:,.2f} kWh")

    model_bundle = load_model_bundle()
    df = load_simulation_data()

    predict_one = make_fast_predictor(model_bundle)
    day_library = build_day_library(df)
    on_pool, off_dry_pool, off_rain_pool = estimate_transition_pools(df)

    moisture_min = float(df["Soil Moisture [RH%]"].min())
    moisture_max = float(df["Soil Moisture [RH%]"].max())
    observed_starts = df["Soil Moisture [RH%]"].to_numpy(dtype=float)

    simulated_water_savings = []
    simulated_energy_savings = []

    rng = np.random.default_rng(42)

    print(
        f"Running {SEASONS_TO_SIMULATE} stateful seasonal simulations "
        f"({TOTAL_SAMPLES} control steps each)..."
    )

    for _ in range(SEASONS_TO_SIMULATE):
        exogenous_season = bootstrap_exogenous_season(day_library, rng)
        moisture_start = float(rng.choice(observed_starts))

        season_water_liters, season_energy_kwh = simulate_one_season(
            predict_one=predict_one,
            exogenous_season=exogenous_season,
            on_pool=on_pool,
            off_dry_pool=off_dry_pool,
            off_rain_pool=off_rain_pool,
            moisture_min=moisture_min,
            moisture_max=moisture_max,
            moisture_start=moisture_start,
            rng=rng,
        )

        water_saving_pct = ((IRRIFRAME_WATER_LITERS - season_water_liters) / IRRIFRAME_WATER_LITERS) * 100.0
        energy_saving_pct = ((IRRIFRAME_ENERGY_KWH - season_energy_kwh) / IRRIFRAME_ENERGY_KWH) * 100.0

        simulated_water_savings.append(water_saving_pct)
        simulated_energy_savings.append(energy_saving_pct)

    simulated_water_savings = np.array(simulated_water_savings, dtype=float)
    simulated_energy_savings = np.array(simulated_energy_savings, dtype=float)

    w_lower, w_upper = confidence_interval_95(simulated_water_savings)
    e_lower, e_upper = confidence_interval_95(simulated_energy_savings)

    print("\n--- Monte Carlo Simulation Results (95% Confidence Interval) ---")
    print(f"Estimated Water Savings:  {w_lower:.2f}% to {w_upper:.2f}%")
    print(f"Estimated Energy Savings: {e_lower:.2f}% to {e_upper:.2f}%")

    print("\nDiagnostic means:")
    print(f"Mean Water Saving:  {np.mean(simulated_water_savings):.2f}%")
    print(f"Mean Energy Saving: {np.mean(simulated_energy_savings):.2f}%")


if __name__ == "__main__":
    run_monte_carlo_simulation()