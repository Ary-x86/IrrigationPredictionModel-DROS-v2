import numpy as np
import pandas as pd

def run_monte_carlo_simulation():
    print("--- Initiating Monte Carlo Simulation ---")
    
    # =========================================================================
    # THE THEORY OF MONTE CARLO
    # A Monte Carlo simulation is a mathematical technique used to estimate the 
    # possible outcomes of an uncertain event. Instead of calculating one "average" 
    # season, it runs hundreds or thousands of randomized simulations using a 
    # probability distribution[cite: 335]. 
    # By running the simulation 1,000 times, the law of large numbers takes over, 
    # giving us a beautiful bell curve of all statistically probable outcomes.
    # =========================================================================

    # 1. Setup the Simulation Parameters based on the paper
    seasons_to_simulate = 1000 # [cite: 101, 334]
    days_per_season = 70       # Conventional irrigation period [cite: 336]
    samples_per_day = 24 * 6   # 10-minute intervals (6 per hour * 24 hours)
    total_samples = days_per_season * samples_per_day # 10,080 samples per season [cite: 336]

    # Hardware specs
    flow_rate_lph = 300        # Liters per hour delivered by the valves [cite: 225, 337]
    liters_per_10min = flow_rate_lph * (10 / 60) # 50 liters per 10-minute activation
    pump_power_kw = 4.0        # The pump driving the system uses 4 kW [cite: 339]

    # Baseline: The conventional recommendation from the IRRIFRAME platform
    # The paper notes IRRIFRAME recommends 324.5 mm for a 0.0132 ha (132 m^2) surface[cite: 82].
    # 1 mm of water on 1 m^2 = 1 Liter. So 324.5 mm on 132 m^2 = 42,834 Liters.
    irriframe_water_liters = 324.5 * 132 
    # To pump 42,834 L at 300 L/hour takes ~142.78 hours. At 4 kW, that's ~571.12 kWh.
    irriframe_energy_kwh = (irriframe_water_liters / flow_rate_lph) * pump_power_kw

    print(f"Baseline IRRIFRAME Water Usage: {irriframe_water_liters:,.2f} Liters")
    #print(f"Baseline IRRIFRAME Energy Usage: {irriframe_energy_kwh:,.2f} kWh\n")


    # # Let's assume we calculated mu and sigma from step 03
    # mu_capacity = 25.0 
    # sigma_capacity = 4.0

    # =========================================================================
    # FIX: Use the actual mu and sigma from your Step 03 output!
    # =========================================================================
    mu_capacity = 27.29    # Your calculated Soil Capacity
    sigma_capacity = 3.05  # Your calculated Standard Deviation

    # # The paper dictates using a normal distribution characterized by the soil capacity 
    # # level as the mean value and (mu - sigma/2) to define the standard deviation parameters[cite: 336].
    # sim_mean = mu_capacity
    # sim_std = mu_capacity - (sigma_capacity / 2)

    # The simulation's spread should be half the measured sigma, 
    # NOT mu - sigma/2. This keeps moisture levels tightly around the capacity point.
    sim_mean = mu_capacity
    sim_std = sigma_capacity / 2  # Corrected Spread (~1.52%)

    # Arrays to store the total savings of each of the 1,000 simulated seasons
    simulated_water_savings = []
    simulated_energy_savings = []


    

    print(f"Running {seasons_to_simulate} simulations with sim_std = {sim_std:.2f}...\tThis might take a moment...")
    #print(f"Running {seasons_to_simulate} seasonal simulations. This might take a moment...")

    # 2. Run the Monte Carlo Loop
    for season in range(seasons_to_simulate):
        # Generate 10,080 random soil moisture states for the season based on our probability distribution [cite: 336]
        simulated_moisture = np.random.normal(loc=sim_mean, scale=sim_std, size=total_samples)
        
        # Determine how many times the AI would turn the water ON.
        # In the paper, ON is triggered when moisture falls below the lower limit[cite: 261].
        # For this simulation, we count how many 10-min intervals drop below our threshold.
        # (Note: In reality, the AI also checks rain forecasts, but this mimics the physical distribution).

        # Count activations (Moisture < mu - sigma/2) [cite: 222, 261]
        threshold = mu_capacity - (sigma_capacity / 2)
        activations = np.sum(simulated_moisture < threshold)
        
        # Calculate consumption for this specific random season [cite: 337]
        season_water_liters = activations * liters_per_10min
        season_pump_hours = (activations * 10) / 60
        season_energy_kwh = season_pump_hours * pump_power_kw
        
        # Calculate percentage savings compared to the IRRIFRAME baseline [cite: 338]
        water_saving_pct = ((irriframe_water_liters - season_water_liters) / irriframe_water_liters) * 100
        energy_saving_pct = ((irriframe_energy_kwh - season_energy_kwh) / irriframe_energy_kwh) * 100
        
        simulated_water_savings.append(water_saving_pct)
        simulated_energy_savings.append(energy_saving_pct)

    # 3. Analyze the Results using a 95% Confidence Interval
    # The paper provides an appropriate confidence interval with 95% coverage probability 
    # by adding/subtracting two estimated standard deviations from the mean.
    
    # mean_water_saving = np.mean(simulated_water_savings)
    # std_water_saving = np.std(simulated_water_savings)
    # water_ci_lower = mean_water_saving - (2 * std_water_saving)
    # water_ci_upper = mean_water_saving + (2 * std_water_saving)

    # mean_energy_saving = np.mean(simulated_energy_savings)
    # std_energy_saving = np.std(simulated_energy_savings)
    # energy_ci_lower = mean_energy_saving - (2 * std_energy_saving)
    # energy_ci_upper = mean_energy_saving + (2 * std_energy_saving)


    # 3. Analyze the Results (95% Confidence Interval) [cite: 341]
    def get_ci(data):
        mean = np.mean(data)
        std = np.std(data)
        return mean - (2 * std), mean + (2 * std)

    w_lower, w_upper = get_ci(simulated_water_savings)
    e_lower, e_upper = get_ci(simulated_energy_savings)

    print("\n--- Monte Carlo Simulation Results (95% Confidence Interval) ---")
    print(f"Estimated Water Savings:  {w_lower:.1f}% to {w_upper:.1f}%")
    print(f"(Paper achieved 14.5% to 27.6%)")
    
    print(f"Estimated Energy Savings: {e_lower:.1f}% to {e_upper:.1f}%")
    print(f"(Paper achieved 49.2% to 57.0%)")

if __name__ == "__main__":
    run_monte_carlo_simulation()