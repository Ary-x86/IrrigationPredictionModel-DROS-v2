import time
import joblib
import pandas as pd
import requests
from datetime import datetime

# ==========================================
# HARDWARE MOCK FUNCTIONS
# In a real deployment, these functions would connect to your LoRaWAN 
# gateway or GPIO pins to read sensors and open physical valves.
# ==========================================

def read_live_sensors():
    """Mocks reading data from the field end nodes."""
    # In reality, this reads your Milesight and Talkpool sensors [cite: 136-137].
    return {
        'Soil Moisture [RH%]': 21.5,           # Currently getting dry
        'Soil Temperature [C]': 24.0,
        'Environmental Temperature [ C]': 28.5,
        'Environmental Humidity [RH %]': 45.0
    }

def control_water_valve(state):
    """Mocks turning the physical irrigation valve ON or OFF."""
    if state == "ON":
        print("🚰 COMMAND: Activating water valve! Delivering 300 L/hr.")
    elif state == "OFF":
        print("🛑 COMMAND: Shutting off water valve.")

def send_farmer_alert(message):
    """Mocks sending an SMS or email alert to the farmer."""
    print(f"⚠️ ALERT DISPATCHED: {message}")

# ==========================================
# WEATHER API FUNCTION
# ==========================================

def fetch_live_forecast():
    """Fetches the 3-day forecast from Open Meteo for the current time."""
    # Using the coordinates from the living lab[cite: 141].
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 44.1125,
        "longitude": 10.411,
        "hourly": "relative_humidity_2m,precipitation,et0_fao_evapotranspiration",
        "timezone": "Europe/Rome",
        "forecast_days": 3
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # We grab the forecast for the immediate next hour
        return {
            'Weather Forecast Rainfall [mm]': data['hourly']['precipitation'][0],
            'Crop Data Evapotranspiration [mm]': data['hourly']['et0_fao_evapotranspiration'][0]
        }
    except Exception as e:
        print(f"Error fetching weather: {e}")
        # Fail-safe defaults if API goes down
        return {'Weather Forecast Rainfall [mm]': 0.0, 'Crop Data Evapotranspiration [mm]': 0.0}

# ==========================================
# MAIN CONTROLLER LOOP
# ==========================================

def run_irrigation_system():
    print("Initializing Smart Irrigation Edge Controller...")
    
    # 1. Load the trained Multi-Layer Perceptron Neural Network
    try:
        model = joblib.load('models/mlp_irrigation_model.pkl')
        print("✅ Neural Network Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: Model not found. Did you run 04_train_neural_network.py first?")
        return

    print("Beginning 10-minute monitoring loop. Press Ctrl+C to exit.\n")

    # 2. The Continuous Edge Computing Loop
    # The paper dictates that signals are transmitted and processed every 10 mins [cite: 131-132].
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"--- System Check: {current_time} ---")

        # Step A: Gather all 6 real-time features required by the AI [cite: 523-524]
        sensor_data = read_live_sensors()
        weather_data = fetch_live_forecast()
        
        # Combine into a single dictionary
        live_features = {**sensor_data, **weather_data}
        
        # Convert to a DataFrame so it matches the format the model was trained on
        df_live = pd.DataFrame([live_features])
        
        print(f"📊 Current Soil Moisture: {live_features['Soil Moisture [RH%]']}%")
        print(f"☁️  Forecasted Rain: {live_features['Weather Forecast Rainfall [mm]']} mm")

        # Step B: Feed data to the Neural Network to get the prediction
        # The outputs are four classes: OFF (0), ON (1), No Adjustment (2), Alert (3) [cite: 260-264, 520].
        prediction = model.predict(df_live)[0]

        # Step C: Execute the physical hardware commands based on the AI's decision [cite: 87, 260-264]
        if prediction == 0:
            print("🤖 AI Decision: Class 0 (OFF). Soil is saturated past capacity.")
            control_water_valve("OFF")
            
        elif prediction == 1:
            print("🤖 AI Decision: Class 1 (ON). Soil is dry and no rain is coming.")
            control_water_valve("ON")
            
        elif prediction == 2:
            print("🤖 AI Decision: Class 2 (No Adjustment). Conditions are optimal.")
            # Do nothing
            
        elif prediction == 3:
            print("🤖 AI Decision: Class 3 (Alert!). Soil is critically dry, but rain is expected.")
            send_farmer_alert("Critically low soil moisture detected, but high rainfall is forecasted. Manual check advised!")

        print("-" * 40)
        
        # Step D: Wait 10 minutes (600 seconds) before checking again [cite: 131-132]
        # For testing purposes, you might want to change this to 5 seconds!
        time.sleep(600) 

if __name__ == "__main__":
    try:
        run_irrigation_system()
    except KeyboardInterrupt:
        print("\nSystem gracefully shut down by user.")