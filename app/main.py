from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Healthcare ML System", version="1.0.0")

# Robust Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load models if they exist
MODELS = {}
MODEL_PATHS = {
    "classification": "models/heart_disease_model.pkl",
    "regression": "models/cost_prediction_model.pkl",
    "clustering": "models/patient_cluster_model.pkl"
}

def load_models():
    # Use CWD for models or fix paths similarly if needed, but keeping simple for now
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
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict/heart-disease")
def predict_heart_disease(patient: PatientData):
    if "classification" not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = MODELS["classification"]
    # Create DataFrame with proper column names to avoid warnings
    input_df = pd.DataFrame([{
        'age': patient.age,
        'bmi': patient.bmi,
        'blood_pressure': patient.blood_pressure,
        'glucose': patient.glucose
    }])
    
    prediction = model.predict(input_df)[0]
    # Check if the model has predict_proba (GradientBoosting does)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
    else:
        probability = float(prediction) # Fallback
    
    return {
        "heart_disease_prediction": int(prediction),
        "risk_probability": float(probability)
    }

@app.post("/predict/cost")
def predict_cost(patient: PatientData):
    if "regression" not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = MODELS["regression"]
    input_df = pd.DataFrame([{
        'age': patient.age,
        'bmi': patient.bmi,
        'blood_pressure': patient.blood_pressure,
        'glucose': patient.glucose
    }])
    
    prediction = model.predict(input_df)[0]
    
    return {"estimated_hospital_cost": float(prediction)}

@app.post("/predict/cluster")
def predict_cluster(patient: PatientData):
    if "clustering" not in MODELS:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    model = MODELS["clustering"]
    input_df = pd.DataFrame([{
        'age': patient.age,
        'bmi': patient.bmi,
        'blood_pressure': patient.blood_pressure,
        'glucose': patient.glucose
    }])
    
    cluster = model.predict(input_df)[0]
    
    return {"patient_segment_cluster": int(cluster)}
