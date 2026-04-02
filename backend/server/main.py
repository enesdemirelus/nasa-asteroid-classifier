import os
import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def remove_unused_columns(df):
    columns_to_drop = [
        "Neo Reference ID", "Name", "Est Dia in M(min)", "Est Dia in M(max)",
        "Est Dia in Miles(min)", "Est Dia in Miles(max)", "Est Dia in Feet(min)",
        "Est Dia in Feet(max)", "Close Approach Date", "Epoch Date Close Approach",
        "Relative Velocity km per hr", "Miss Dist.(Astronomical)", "Miss Dist.(lunar)",
        "Miss Dist.(miles)", "Orbit ID", "Orbit Determination Date", "Orbiting Body",
        "Equinox", "Epoch Osculation", "Miles per hour",
    ]
    return df.drop(columns=columns_to_drop, errors="ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, "model.keras"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

class AsteroidInput(BaseModel):
    data: dict

@app.post("/predict")
def predict(input: AsteroidInput):
    df = pd.DataFrame([input.data])
    cleaned = remove_unused_columns(df)
    features = cleaned.drop(columns=["Hazardous"], errors="ignore")
    features = features.reindex(columns=feature_columns)
    scaled = scaler.transform(features)
    probability = float(model.predict(scaled, verbose=0).flatten()[0])
    return {
        "hazard_probability": probability,
        "predicted_hazardous": probability >= 0.5,
    }
