from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Healthcare ML System", version="1.0.0")

# Load models if they exist
MODELS = {}
MODEL_PATHS = {
    "classification": "models/heart_disease_model.pkl",
    "regression": "models/cost_prediction_model.pkl",
    "clustering": "models/patient_cluster_model.pkl"
}

def load_models():
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            MODELS[name] = joblib.load(path)
        else:
            print(f"Warning: Model {name} not found at {path}")

# Load on startup
load_models()

class PatientData(BaseModel):
    age: int
    bmi: float
    blood_pressure: float
    glucose: float

@app.get("/")
def read_root():
    return {"message": "Healthcare Prediction API Ready"}

@app.post("/predict/heart-disease")
def predict_heart_disease(patient: PatientData):
    if "classification" not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = MODELS["classification"]
    input_data = [[patient.age, patient.bmi, patient.blood_pressure, patient.glucose]]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "heart_disease_prediction": int(prediction),
        "risk_probability": float(probability)
    }

@app.post("/predict/cost")
def predict_cost(patient: PatientData):
    if "regression" not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = MODELS["regression"]
    input_data = [[patient.age, patient.bmi, patient.blood_pressure, patient.glucose]]
    prediction = model.predict(input_data)[0]
    
    return {"estimated_hospital_cost": float(prediction)}

@app.post("/predict/cluster")
def predict_cluster(patient: PatientData):
    if "clustering" not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    model = MODELS["clustering"]
    input_data = [[patient.age, patient.bmi, patient.blood_pressure, patient.glucose]]
    cluster = model.predict(input_data)[0]
    
    return {"patient_segment_cluster": int(cluster)}
