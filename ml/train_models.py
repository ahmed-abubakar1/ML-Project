import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import joblib
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    # Synthetic Healthcare Data
    # Age, BMI, BloodPressure, Glucose, SufferHeartDisease (Target 1), HospitalCost (Target 2)
    
    data = {
        'age': np.random.randint(20, 90, n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'blood_pressure': np.random.normal(120, 15, n_samples),
        'glucose': np.random.normal(100, 20, n_samples),
    }
    df = pd.DataFrame(data)
    
    # Target 1: Classification (Heart Disease)
    # Simple logic: Age > 50 and BMI > 30 increases risk
    risk_score = (df['age'] / 90) + (df['bmi'] / 40) + np.random.normal(0, 0.1, n_samples)
    df['heart_disease'] = (risk_score > 1.2).astype(int)
    
    # Target 2: Regression (Hospital Cost)
    # Cost correlated with Age and some random noise
    df['hospital_cost'] = 1000 + (df['age'] * 50) + (df['heart_disease'] * 5000) + np.random.normal(0, 500, n_samples)
    
    return df

def train_models():
    print("Generating data...")
    df = generate_synthetic_data()
    df.to_csv("data/healthcare_data.csv", index=False)
    
    X = df[['age', 'bmi', 'blood_pressure', 'glucose']]
    y_class = df['heart_disease']
    y_reg = df['hospital_cost']
    
    # 1. Classification Model
    print("Training Classification Model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_class)
    joblib.dump(clf, "models/heart_disease_model.pkl")
    
    # 2. Regression Model
    print("Training Regression Model...")
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y_reg)
    joblib.dump(reg, "models/cost_prediction_model.pkl")
    
    # 3. Clustering (Unsupervised)
    print("Training Clustering Model...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    joblib.dump(kmeans, "models/patient_cluster_model.pkl")
    
    print("Models saved in 'models/' directory.")

if __name__ == "__main__":
    train_models()
