# src/04_train_neural_network.py

#see the v1 for monte carlo explanations
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


FEATURE_COLUMNS = [
    "Soil Moisture [RH%]",
    "Soil Temperature [C]",
    "Environmental Temperature [ C]",
    "Environmental Humidity [RH %]",
    "Weather Forecast Rainfall [mm]",
    "Crop Data Evapotranspiration [mm]",
]


def oversample_training_set(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simple bootstrap oversampling to reduce the chance that the rare Alert class is ignored.
    This is applied ONLY to the training split, never to the test split.
    """
    df_train = X_train.copy()
    df_train["__target__"] = y_train.values

    counts = df_train["__target__"].value_counts()
    max_count = counts.max()

    balanced_parts = []
    for class_label in counts.index:
        class_rows = df_train[df_train["__target__"] == class_label]
        balanced_parts.append(
            class_rows.sample(n=max_count, replace=True, random_state=42)
        )

    df_balanced = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    X_balanced = df_balanced.drop(columns="__target__")
    y_balanced = df_balanced["__target__"]

    return X_balanced, y_balanced


def build_model() -> Pipeline:
    """
    Keep the paper's final MLP structure and hyperparameters,
    but wrap it in a scaler for better numerical stability.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(6, 12, 6),
                    activation="tanh",
                    solver="adam",
                    learning_rate="constant",
                    alpha=0.01,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def train_and_evaluate_model():
    print("Loading processed dataset...")
    df = pd.read_csv(DATA_DIR / "processed_dataset.csv")

    X = df[FEATURE_COLUMNS].copy()
    y = df["Irrigation_Decision"].astype(int).copy()

    print("Class distribution in the full dataset:")
    print(y.value_counts().sort_index())

    print("Splitting data into training and testing sets (stratified 70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print("Oversampling minority classes inside the training set...")
    X_train_balanced, y_train_balanced = oversample_training_set(X_train, y_train)

    print("Building the MLP pipeline...")
    model = build_model()

    print("Training the Artificial Neural Network...")
    model.fit(X_train_balanced, y_train_balanced)
    print("Training complete!")

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.4f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=["OFF (0)", "ON (1)", "No Adj (2)", "Alert (3)"],
            zero_division=0,
        )
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save a bundle instead of a bare model so inference scripts always know the feature order.
    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "notes": "Pipeline(StandardScaler -> MLPClassifier) with paper-aligned MLP hyperparameters.",
    }

    model_path = MODELS_DIR / "mlp_irrigation_model.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\nSuccessfully saved trained model bundle to {model_path}")


if __name__ == "__main__":
    train_and_evaluate_model()