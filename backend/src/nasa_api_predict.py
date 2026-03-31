import requests
import os
import joblib
from dotenv import load_dotenv
import pandas as pd
import tensorflow as tf

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

NASA_API_KEY = os.getenv("NASA_API_KEY")


def load_saved_artifacts():
    model = tf.keras.models.load_model(os.path.join(OUTPUTS_DIR, "model.keras"))
    scaler = joblib.load(os.path.join(OUTPUTS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(OUTPUTS_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns


def prepare_features(asteroid_row, feature_columns, scaler):
    asteroid_df = pd.DataFrame([dict(asteroid_row)])
    features = asteroid_df.reindex(columns=feature_columns)

    missing_columns = [column for column in feature_columns if pd.isna(features.iloc[0][column])]
    if missing_columns:
        raise ValueError(f"Missing required feature values: {missing_columns}")

    return scaler.transform(features)

def get_asteroid_ids_by_date(start_date, end_date):
    url = "https://api.nasa.gov/neo/rest/v1/feed"
    params = {"start_date": start_date, "end_date": end_date, "api_key": NASA_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    near_earth_objects = data["near_earth_objects"]
    return [
        obj["id"]
        for date in near_earth_objects
        for obj in near_earth_objects[date]
    ]

def get_complete_asteroid_data(asteroid_id):
    url = f"https://api.nasa.gov/neo/rest/v1/neo/{asteroid_id}"
    params = {"api_key": NASA_API_KEY}
    response = requests.get(url, params=params)
    res = response.json()
    
    approach = res["close_approach_data"][0]
    orbit = res["orbital_data"]
    
    return {
        "Name": res.get("name"),
        "Absolute Magnitude": float(res.get("absolute_magnitude_h")),
        "Est Dia in KM(min)": float(res["estimated_diameter"]["kilometers"]["estimated_diameter_min"]),
        "Est Dia in KM(max)": float(res["estimated_diameter"]["kilometers"]["estimated_diameter_max"]),
        "Relative Velocity km per sec": float(approach["relative_velocity"]["kilometers_per_second"]),
        "Miss Dist.(kilometers)": float(approach["miss_distance"]["kilometers"]),
        "Orbit Uncertainity": int(orbit["orbit_uncertainty"]),
        "Minimum Orbit Intersection": float(orbit["minimum_orbit_intersection"]),
        "Jupiter Tisserand Invariant": float(orbit["jupiter_tisserand_invariant"]),
        "Eccentricity": float(orbit["eccentricity"]),
        "Semi Major Axis": float(orbit["semi_major_axis"]),
        "Inclination": float(orbit["inclination"]),
        "Asc Node Longitude": float(orbit["ascending_node_longitude"]),
        "Orbital Period": float(orbit["orbital_period"]),
        "Perihelion Distance": float(orbit["perihelion_distance"]),
        "Perihelion Arg": float(orbit["perihelion_argument"]),
        "Aphelion Dist": float(orbit["aphelion_distance"]),
        "Perihelion Time": float(orbit["perihelion_time"]),
        "Mean Anomaly": float(orbit["mean_anomaly"]),
        "Mean Motion": float(orbit["mean_motion"]),
        "Hazardous": res["is_potentially_hazardous_asteroid"]
    }


def predict_asteroid(asteroid_data, model, scaler, feature_columns):
    scaled_features = prepare_features(asteroid_data, feature_columns, scaler)
    probability = float(model.predict(scaled_features, verbose=0).flatten()[0])
    prediction = probability >= 0.5

    result = dict(asteroid_data)
    result["Predicted Hazardous"] = prediction
    result["Hazard Probability"] = probability
    return result


if __name__ == "__main__":
    start_date = "2026-03-30"
    end_date = "2026-03-31"
    model, scaler, feature_columns = load_saved_artifacts()
    id_list = get_asteroid_ids_by_date(start_date, end_date)
    
    for a_id in id_list:
        full_info = get_complete_asteroid_data(a_id)
        prediction = predict_asteroid(full_info, model, scaler, feature_columns)
        print(f"--- {prediction['Name']} ({a_id}) ---")
        print(f"Actual Hazardous: {prediction['Hazardous']}")
        print(f"Predicted Hazardous: {prediction['Predicted Hazardous']}")
        print(f"Hazard Probability: {prediction['Hazard Probability']:.4f}")
        print("\n" + "=" * 40 + "\n")