import os
import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from train import remove_unused_columns

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
