import os

import joblib
import pandas as pd
import tensorflow as tf

from train import remove_unused_columns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_PATH = os.path.join(BASE_DIR, "../dataset/nasa.csv")
SAMPLE_SIZE = 5


def load_saved_artifacts():
    model = tf.keras.models.load_model(os.path.join(OUTPUTS_DIR, "model.keras"))
    scaler = joblib.load(os.path.join(OUTPUTS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(OUTPUTS_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns


def load_random_asteroids(sample_size=SAMPLE_SIZE):
    df = pd.read_csv(DATASET_PATH)
    return df.sample(n=sample_size).reset_index(drop=True)


def prepare_features(sample_df, feature_columns, scaler):
    cleaned_df = remove_unused_columns(sample_df.copy())
    features = cleaned_df.drop(columns=["Hazardous"])
    features = features.reindex(columns=feature_columns)
    return scaler.transform(features)


def predict_asteroids():
    model, scaler, feature_columns = load_saved_artifacts()
    sampled_asteroids = load_random_asteroids()
    scaled_features = prepare_features(sampled_asteroids, feature_columns, scaler)

    probabilities = model.predict(scaled_features, verbose=0).flatten()
    predictions = probabilities >= 0.5

    results_df = pd.DataFrame(
        {
            "Neo Reference ID": sampled_asteroids["Neo Reference ID"].astype(str),
            "Actual Hazardous": sampled_asteroids["Hazardous"].astype(bool),
            "Predicted Hazardous": predictions,
            "Hazard Probability": probabilities,
        }
    )

    return results_df

if __name__ == "__main__":
    prediction_results = predict_asteroids()
    print(prediction_results.to_string(index=False))
