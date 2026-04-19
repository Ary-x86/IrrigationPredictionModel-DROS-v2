# src/05_monte_carlo_simulation.py

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

SEASONS_TO_SIMULATE = 1000
DAYS_PER_SEASON = 70
SAMPLES_PER_DAY = 24 * 6
TOTAL_SAMPLES = DAYS_PER_SEASON * SAMPLES_PER_DAY  # 10,080

FLOW_RATE_LPH = 300.0
LITERS_PER_10MIN = FLOW_RATE_LPH * (10.0 / 60.0)   # 50 L
PUMP_POWER_KW = 4.0
KWH_PER_10MIN = PUMP_POWER_KW * (10.0 / 60.0)

IRRIFRAME_MM = 324.5
AREA_M2 = 132.0
IRRIFRAME_WATER_LITERS = IRRIFRAME_MM * AREA_M2
IRRIFRAME_ENERGY_KWH = (IRRIFRAME_WATER_LITERS / FLOW_RATE_LPH) * PUMP_POWER_KW


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    lower = mean - (2.0 * std)
    upper = mean + (2.0 * std)

    # Physical upper bound: you cannot save more than 100%
    upper = min(upper, 100.0)
    return lower, upper


def run_monte_carlo_simulation():
    print("--- Initiating Monte Carlo Simulation ---")

    with open(DATA_DIR / "soil_capacity_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df = pd.read_csv(DATA_DIR / "processed_dataset.csv")

    bundle = joblib.load(MODELS_DIR / "mlp_irrigation_model.pkl")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    mu = float(metadata["mu"])
    sigma = float(metadata["sigma"])

    # IMPORTANT:
    # The paper text says "mu - sigma/2 as the standard deviation", which is not
    # numerically stable if taken literally. So we make the Monte Carlo spread explicit.
    # A narrow spread around the capacity point is the practical interpretation.
    mc_soil_std = sigma / 3.0

    soil_min = float(df["Soil Moisture [RH%]"].min())
    soil_max = float(df["Soil Moisture [RH%]"].max())

    # Keep the other five inputs fixed at their empirical means.
    # This is the cleanest reconstruction of a paper-style classifier-level Monte Carlo.
    feature_means = df[feature_columns].mean()

    rng = np.random.default_rng(42)

    water_savings = []
    energy_savings = []

    print(f"Baseline IRRIFRAME Water Usage:  {IRRIFRAME_WATER_LITERS:,.2f} Liters")
    print(f"Baseline IRRIFRAME Energy Usage: {IRRIFRAME_ENERGY_KWH:,.2f} kWh")
    print(
        f"Running {SEASONS_TO_SIMULATE} classifier-level Monte Carlo seasons "
        f"({TOTAL_SAMPLES} samples each, mc_soil_std={mc_soil_std:.4f})..."
    )

    for _ in range(SEASONS_TO_SIMULATE):
        simulated_moisture = rng.normal(loc=mu, scale=mc_soil_std, size=TOTAL_SAMPLES)
        simulated_moisture = np.clip(simulated_moisture, soil_min, soil_max)

        X_sim = pd.DataFrame(index=range(TOTAL_SAMPLES), columns=feature_columns, dtype=float)

        for col in feature_columns:
            X_sim[col] = float(feature_means[col])

        X_sim["Soil Moisture [RH%]"] = simulated_moisture

        predictions = model.predict(X_sim)

        # Paper-faithful reading: each class-1 output is a 10-minute activation input.
        on_activations = int(np.sum(predictions == 1))

        season_water_liters = on_activations * LITERS_PER_10MIN
        season_energy_kwh = on_activations * KWH_PER_10MIN

        water_saving_pct = ((IRRIFRAME_WATER_LITERS - season_water_liters) / IRRIFRAME_WATER_LITERS) * 100.0
        energy_saving_pct = ((IRRIFRAME_ENERGY_KWH - season_energy_kwh) / IRRIFRAME_ENERGY_KWH) * 100.0

        water_savings.append(water_saving_pct)
        energy_savings.append(energy_saving_pct)

    water_savings = np.array(water_savings, dtype=float)
    energy_savings = np.array(energy_savings, dtype=float)

    w_lower, w_upper = confidence_interval_95(water_savings)
    e_lower, e_upper = confidence_interval_95(energy_savings)

    print("\n--- Monte Carlo Simulation Results (95% Confidence Interval) ---")
    print(f"Estimated Water Savings:  {w_lower:.2f}% to {w_upper:.2f}%")
    print(f"Estimated Energy Savings: {e_lower:.2f}% to {e_upper:.2f}%")

    print("\nDiagnostic means:")
    print(f"Mean Water Saving:  {np.mean(water_savings):.2f}%")
   # print(f"Mean Energy Saving: {np.mean(energy_savings):.2f}%")

if __name__ == "__main__":
    run_monte_carlo_simulation()

# src/05_monte_carlo_simulation.py
    """
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

# =========================================================================
# THE THEORY OF MONTE CARLO
# A Monte Carlo simulation is a mathematical technique used to estimate the
# possible outcomes of an uncertain event. Instead of calculating one "average"
# season, it runs hundreds or thousands of randomized simulations using a
# probability distribution.
# By running the simulation many times, the law of large numbers takes over,
# giving us a distribution of statistically probable outcomes.
# =========================================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

# Paper-aligned season setup.
# For debugging, start lower. Once results look sane, raise it.
SEASONS_TO_SIMULATE = 200
DAYS_PER_SEASON = 70
SAMPLES_PER_DAY = 24 * 6  # 10-minute control loop
TOTAL_SAMPLES = DAYS_PER_SEASON * SAMPLES_PER_DAY

# Irrigation hardware.
FLOW_RATE_LPH = 300.0
LITERS_PER_10MIN = FLOW_RATE_LPH * (10.0 / 60.0)  # 50 L
PUMP_POWER_KW = 4.0
KWH_PER_10MIN = PUMP_POWER_KW * (10.0 / 60.0)

# IRRIFRAME baseline from the paper.
IRRIFRAME_MM = 324.5
AREA_M2 = 132.0
IRRIFRAME_WATER_LITERS = IRRIFRAME_MM * AREA_M2
IRRIFRAME_ENERGY_KWH = (IRRIFRAME_WATER_LITERS / FLOW_RATE_LPH) * PUMP_POWER_KW

# Prevent pathological ON-ON-ON spam when one irrigation pulse was just triggered.
COOLDOWN_STEPS = 1

# Performance controls.
# Bigger chunk = faster, but more approximation from the provisional trajectory.
# 500-1500 is a good range. 720 = 5 days.
CHUNK_SIZE = 720

# Debug / UX
DEBUG_FAST_MODE = False      # If True, force a very small number of seasons
DEBUG_SEASONS = 20
PROGRESS_EVERY = 10          # Print progress every N seasons

# Safety cap: if a season somehow becomes absurdly wet, clamp moisture there.
# These are soft caps derived from observed historical values anyway.
MIN_ON_DELTA = 0.05          # Ensure one ON pulse raises moisture at least a little


# -------------------------------------------------------------------------
# LOADING
# -------------------------------------------------------------------------

def load_model_bundle():
    model_path = MODELS_DIR / "mlp_irrigation_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    return joblib.load(model_path)


def load_simulation_data():
    data_path = DATA_DIR / "modeling_dataset_full.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Simulation dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# -------------------------------------------------------------------------
# PREP
# -------------------------------------------------------------------------

def build_day_library(df: pd.DataFrame) -> list[np.ndarray]:
    """
    # Build a library of daily exogenous-condition blocks.

    # Each block has 144 rows x 5 columns:
    #     soil_temp, env_temp, env_humidity, rainfall, evapotranspiration

    # We deliberately keep moisture OUT of the templates because moisture is stateful
    # and must be simulated, not copied from history.

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
    # Build empirical one-step moisture-change pools from the historical dataset.

    # We estimate:
    # - on_pool: moisture changes observed when irrigation should be ON
    # - off_dry_pool: valve closed + no rain
    # - off_rain_pool: valve closed + rain present

    # This gives the Monte Carlo a state transition model instead of fake iid threshold counting.
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
        on_pool = np.array([0.10], dtype=float)
    if len(off_dry_pool) == 0:
        off_dry_pool = np.array([-0.05], dtype=float)
    if len(off_rain_pool) == 0:
        off_rain_pool = off_dry_pool.copy()

    return on_pool, off_dry_pool, off_rain_pool


def bootstrap_exogenous_season(day_library: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    # Build a 70-day season by sampling daily exogenous blocks with replacement.
    # This preserves realistic intra-day weather/temperature structure better than iid row sampling.
    """
    season_blocks = []
    for _ in range(DAYS_PER_SEASON):
        block = day_library[rng.integers(0, len(day_library))]
        season_blocks.append(block)
    return np.vstack(season_blocks)


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean - (2.0 * std), mean + (2.0 * std)


# -------------------------------------------------------------------------
# MODEL PREDICTION
# -------------------------------------------------------------------------

def predict_batch(pipeline, feature_columns: list[str], feature_matrix: np.ndarray) -> np.ndarray:
    """
    # Predict decisions for many timesteps at once using a DataFrame with the correct
    # feature names. This avoids the sklearn warning about invalid feature names.
    """
    X_df = pd.DataFrame(feature_matrix, columns=feature_columns)
    return pipeline.predict(X_df).astype(int)


# -------------------------------------------------------------------------
# SEASON SIMULATION
# -------------------------------------------------------------------------

def simulate_one_season_chunked(
    pipeline,
    feature_columns: list[str],
    exogenous_season: np.ndarray,
    on_pool: np.ndarray,
    off_dry_pool: np.ndarray,
    off_rain_pool: np.ndarray,
    moisture_min: float,
    moisture_max: float,
    moisture_start: float,
    rng: np.random.Generator,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[float, float]:
    """
    # Stateful 10-minute simulation with chunked model predictions.

    # Why this exists:
    # - A fully stepwise approach is very accurate but too slow because it does one
    #   model prediction per timestep.
    # - A fully vectorized approach would break realism because current moisture depends
    #   on previous decisions.

    # So this function uses a hybrid:
    # 1) Build a provisional feature matrix for a chunk using a no-irrigation drift path.
    # 2) Predict all decisions for that chunk in ONE pipeline call.
    # 3) Apply those decisions step-by-step for the true state evolution.

    # This is much faster while keeping the simulation stateful.
    """
    current_moisture = float(moisture_start)
    total_water_liters = 0.0
    total_energy_kwh = 0.0
    cooldown = 0

    n = len(exogenous_season)
    idx = 0

    while idx < n:
        end = min(idx + chunk_size, n)
        chunk = exogenous_season[idx:end]
        chunk_len = len(chunk)

        # -------------------------------------------------------------
        # PASS 1: Build a provisional feature matrix for batch prediction
        # -------------------------------------------------------------
        # We use a temporary moisture trajectory that assumes no irrigation.
        # This is not perfect, but it is a good approximation that lets us batch
        # model calls instead of predicting one row at a time.
        provisional_features = np.zeros((chunk_len, 6), dtype=float)
        temp_moisture = current_moisture

        for j, row in enumerate(chunk):
            soil_temp, env_temp, env_humidity, rain_mm, et_mm = row

            provisional_features[j] = [
                temp_moisture,
                soil_temp,
                env_temp,
                env_humidity,
                rain_mm,
                et_mm,
            ]

            # Provisional drift assumes valve remains closed.
            if rain_mm > 0:
                drift_delta = float(rng.choice(off_rain_pool))
            else:
                drift_delta = float(rng.choice(off_dry_pool))

            temp_moisture = float(np.clip(temp_moisture + drift_delta, moisture_min, moisture_max))

        # -------------------------------------------------------------
        # PASS 2: Batch predict the chunk in one shot
        # -------------------------------------------------------------
        decisions = predict_batch(pipeline, feature_columns, provisional_features)

        # -------------------------------------------------------------
        # PASS 3: Apply decisions for the real state evolution
        # -------------------------------------------------------------
        for j, row in enumerate(chunk):
            soil_temp, env_temp, env_humidity, rain_mm, et_mm = row
            decision = int(decisions[j])

            # Optional one-step cooldown to stop unrealistic ON spam.
            if cooldown > 0 and decision == 1:
                decision = 2

            if decision == 1:
                total_water_liters += LITERS_PER_10MIN
                total_energy_kwh += KWH_PER_10MIN

                delta = float(rng.choice(on_pool))

                # Ensure that an ON pulse actually raises moisture at least a little.
                delta = max(delta, MIN_ON_DELTA)
                cooldown = COOLDOWN_STEPS
            else:
                if rain_mm > 0:
                    delta = float(rng.choice(off_rain_pool))
                else:
                    delta = float(rng.choice(off_dry_pool))

                cooldown = max(0, cooldown - 1)

            current_moisture = float(np.clip(current_moisture + delta, moisture_min, moisture_max))

        idx = end

    return total_water_liters, total_energy_kwh


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def run_monte_carlo_simulation():
    print("--- Initiating Monte Carlo Simulation ---")
    print(f"Baseline IRRIFRAME Water Usage:  {IRRIFRAME_WATER_LITERS:,.2f} Liters")
    print(f"Baseline IRRIFRAME Energy Usage: {IRRIFRAME_ENERGY_KWH:,.2f} kWh")

    seasons = DEBUG_SEASONS if DEBUG_FAST_MODE else SEASONS_TO_SIMULATE

    model_bundle = load_model_bundle()
    pipeline = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    df = load_simulation_data()

    day_library = build_day_library(df)
    on_pool, off_dry_pool, off_rain_pool = estimate_transition_pools(df)

    moisture_min = float(df["Soil Moisture [RH%]"].min())
    moisture_max = float(df["Soil Moisture [RH%]"].max())
    observed_starts = df["Soil Moisture [RH%]"].to_numpy(dtype=float)

    simulated_water_savings = []
    simulated_energy_savings = []

    rng = np.random.default_rng(42)

    print(
        f"Running {seasons} stateful seasonal simulations "
        f"({TOTAL_SAMPLES} control steps each, chunk_size={CHUNK_SIZE})..."
    )

    t0 = time.time()

    for i in range(seasons):
        exogenous_season = bootstrap_exogenous_season(day_library, rng)
        moisture_start = float(rng.choice(observed_starts))

        season_water_liters, season_energy_kwh = simulate_one_season_chunked(
            pipeline=pipeline,
            feature_columns=feature_columns,
            exogenous_season=exogenous_season,
            on_pool=on_pool,
            off_dry_pool=off_dry_pool,
            off_rain_pool=off_rain_pool,
            moisture_min=moisture_min,
            moisture_max=moisture_max,
            moisture_start=moisture_start,
            rng=rng,
            chunk_size=CHUNK_SIZE,
        )

        water_saving_pct = ((IRRIFRAME_WATER_LITERS - season_water_liters) / IRRIFRAME_WATER_LITERS) * 100.0
        energy_saving_pct = ((IRRIFRAME_ENERGY_KWH - season_energy_kwh) / IRRIFRAME_ENERGY_KWH) * 100.0

        simulated_water_savings.append(water_saving_pct)
        simulated_energy_savings.append(energy_saving_pct)

        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == seasons:
            elapsed = time.time() - t0
            per_season = elapsed / (i + 1)
            remaining = per_season * (seasons - (i + 1))
            print(
                f"Progress: {i + 1}/{seasons} seasons | "
                f"elapsed: {elapsed:.1f}s | "
                f"avg/season: {per_season:.2f}s | "
                f"ETA: {remaining:.1f}s"
            )

    simulated_water_savings = np.array(simulated_water_savings, dtype=float)
    simulated_energy_savings = np.array(simulated_energy_savings, dtype=float)

    w_lower, w_upper = confidence_interval_95(simulated_water_savings)
    e_lower, e_upper = confidence_interval_95(simulated_energy_savings)

    total_elapsed = time.time() - t0

    print("\n--- Monte Carlo Simulation Results (95% Confidence Interval) ---")
    print(f"Estimated Water Savings:  {w_lower:.2f}% to {w_upper:.2f}%")
    print(f"Estimated Energy Savings: {e_lower:.2f}% to {e_upper:.2f}%")

    print("\nDiagnostic means:")
    print(f"Mean Water Saving:  {np.mean(simulated_water_savings):.2f}%")
    print(f"Mean Energy Saving: {np.mean(simulated_energy_savings):.2f}%")

    print("\nRuntime:")
    print(f"Total elapsed time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    run_monte_carlo_simulation()
"""