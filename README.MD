## Initial test for DROS: Making a MLP ANN for smart-irrigation. 

Based on: [](https://www.sciencedirect.com/science/article/pii/S0168169924005179)

We use the "Adam" solver, a constant learning rate, and a regularization index of 0.01. It should have 3 hidden layers containing 6, 12, and 6 neurons respectively. It will use the hyperbolic tangent (tanh) activation function for the hidden layers and a Softmax function for the final output layer.


### Folder LaIt:

```text
smart_irrigation_project/
│
├── data/
│   ├── raw_sensor_data.csv
│   └── processed_dataset.csv
│
├── src/
│   ├── 01_weather_api_fetcher.py
│   ├── 02_data_preprocessing.py
│   ├── 03_soil_capacity_calculator.py
│   ├── 04_train_neural_network.py
│   └── 05_monte_carlo_simulation.py
│
├── models/
│   └── mlp_irrigation_model.pkl
│
├── requirements.txt
└── main_controller.py

```

### The Breakdown & Justification

Here is exactly what goes into each part of the project folder and why the paper requires it:

**1. The `data/` Directory**

* 
**`raw_sensor_data.csv`**: This is where I will temporarily store the raw data coming in from the IoT LoRaWAN network nodes (soil moisture, soil temperature, ambient temperature, ambient humidity, etc.) .


* **`processed_dataset.csv`**: After cleaning the data, I will save it here. The researchers ultimately built a clean dataset of 16,187 samples with 6 specific features to train the AI.



**2. The `src/` Directory (the Codebase)**

* 
**`01_weather_api_fetcher.py`**: The paper relies on a three-day weather forecast to predict rain and evapotranspiration . This script will handle the calls to the open-source Open Meteo API for the specific coordinates.


* **`02_data_preprocessing.py`**: Raw data is messy. I will use the Python `pandas` library in this script to scrub errors, remove duplicates, and handle missing data . This script will also run Pearson and Spearman correlation analyses to drop redundant parameters and avoid overfitting .


* **`03_soil_capacity_calculator.py`**: This is the heart of the "smart" logic. I need a script to calculate the time derivative of the soil moisture to experimentally evaluate the soil capacity point. It will also calculate the statistical mean and standard deviation to build the confidence interval .


* 
**`04_train_neural_network.py`**: This script uses `scikit-learn` to build, train, and test the Artificial Neural Network. It will split the `processed_dataset.csv` into a 70% training and 30% testing set. It will use the "Adam" solver, hyperbolic tangent activation, and a 3-layer architecture to output the four irrigation states .


* **`05_monte_carlo_simulation.py`**: This is an evaluation script. The researchers used a Monte Carlo simulation running 1,000 simulated tomato seasons to mathematically prove the water and energy savings of their AI model.



**3. The `models/` Directory**

* **`mlp_irrigation_model.pkl`**: Once the neural network is trained in script `04`, I will save the actual trained model file here. This allows the live system to make predictions instantly without having to retrain the AI from scratch every time.

**4. The Root Files**

* **`requirements.txt`**: I will list the Python dependencies here so the environment is reproducible. Based on the paper, I will definitely need `pandas` for data handling, `scikit-learn` for the AI, and `mlxtend` to visualize decision regions.


* **`main_controller.py`**: This will be the master script. It will run continuously, fetching API data, reading live sensor data, passing it to the saved AI model, and deciding whether to turn the water valves ON, OFF, do nothing, or send an Alert .


