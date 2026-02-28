#src/04_train_neural_network.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_and_evaluate_model():
    print("Loading processed dataset...")
    
    # 1. Load the finalized dataset
    df = pd.read_csv('./data/processed_dataset.csv')

    # Define the 6 specific features used for predictions [cite: 523, 524]
    X = df[[
        'Soil Moisture [RH%]', 
        'Soil Temperature [C]', 
        'Environmental Temperature [ C]', 
        'Environmental Humidity [RH %]',
        'Weather Forecast Rainfall [mm]',
        'Crop Data Evapotranspiration [mm]'
    ]]
    
    # Define the target classification labels (0: OFF, 1: ON, 2: No Adjustment, 3: Alert)
    y = df['Irrigation_Decision']

    print("Splitting data into training and testing sets...")
    # 2. Train/Test Split
    # The paper explicitly states the database was split 70% for training and 30% for testing[cite: 267].
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Note on Standardization: 
    # The paper notes that performance working with the original dataset was almost identical to the z-score normalized dataset[cite: 269].
    # Therefore, the data were processed in the non-normalized format to give predictions directly in a meaningful measurement[cite: 270].

    print("Building the Multi-Layer Perceptron Neural Network...")
    # 3. Model Configuration
    # We configure the MLPClassifier using the exact hyperparameters from the paper.
    mlp_model = MLPClassifier(
        # The hidden layer configuration resulted from the cross-validation step.
        # It is a three-layer layout with six, twelve, and six neurons[cite: 297].
        hidden_layer_sizes=(6, 12, 6),
        
        # Hyperbolic tangent (tanh) was the corresponding activation function for the hidden layers[cite: 298, 422].
        # (Note: scikit-learn automatically applies the Softmax function to the output layer for multi-class classification [cite: 299]).
        activation='tanh',
        
        # Adaptive moment estimation (adam) was the algorithm used to train the classifier[cite: 294, 422].
        solver='adam',
        
        # The initial learning rate controlled the weight updating, set to a constant value[cite: 304, 422].
        learning_rate='constant',
        
        # Fine tuning of the alpha parameter was crucial to avoid overfitting; the regulation index was 0.01[cite: 303, 422].
        alpha=0.01,
        
        # Add a high max_iter to ensure the model converges during training
        max_iter=2000,
        random_state=42
    )

    print("Training the Artificial Neural Network...")
    # 4. Train the model
    mlp_model.fit(X_train, y_train)
    print("Training complete!")

    print("\n--- Model Evaluation ---")     # 5. Model Evaluation
    # Prediction performance is evaluated by calculating various indicators, such as accuracy, precision, recall, and F1-score[cite: 96, 308].

    y_pred = mlp_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}% (Paper achieved 99.61%)")
    
    # We add labels=[0, 1, 2, 3] to force the report to include all classes
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))
    
    print("\nDetailed Classification Report:")
    # This report automatically calculates the precision, recall, F1-score, macro-average, and weighted-average precision required by the paper[cite: 329, 330].
    print(classification_report(
        y_test, 
        y_pred, 
        labels=[0, 1, 2, 3], 
        target_names=['OFF (0)', 'ON (1)', 'No Adj (2)', 'Alert (3)'],
        zero_division=0 # This prevents a crash if a class has 0 samples
    ))
    


    # 6. Save the Model
    # We save the trained model to the 'models/' directory so the master controller can use it for real-time predictions.
    model_path = './models/mlp_irrigation_model.pkl'
    joblib.dump(mlp_model, model_path)
    print(f"\nSuccessfully saved trained model to {model_path}")

if __name__ == "__main__":
    train_and_evaluate_model()