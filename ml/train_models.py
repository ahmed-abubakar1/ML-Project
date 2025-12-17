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

def generate_synthetic_data():
    print("Fetching 'Live' data from remote source...")
    # Using Pima Indians Diabetes Dataset as a real-world proxy for health vitals
    # Source: https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset
    url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    
    try:
        df = pd.read_csv(url)
        print("Data fetched successfully.")
    except Exception as e:
        print(f"Failed to fetch data: {e}. Falling back to synthetic.")
        # Fallback (simplified)
        return pd.DataFrame({
            'age': np.random.randint(20, 90, 100), 'bmi': np.random.normal(25, 5, 100),
            'blood_pressure': np.random.normal(120, 15, 100), 'glucose': np.random.normal(100, 20, 100),
            'heart_disease': np.random.randint(0, 2, 100), 'hospital_cost': np.random.normal(5000, 1000, 100)
        })

    # Map real columns to our schema
    # Dataset cols: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    df_clean = df.rename(columns={
        'Age': 'age',
        'BMI': 'bmi',
        'BloodPressure': 'blood_pressure',
        'Glucose': 'glucose',
        'Outcome': 'heart_disease' # Using Diabetes Outcome as our Disease Target
    })
    
    # Select only relevant features
    df_clean = df_clean[['age', 'bmi', 'blood_pressure', 'glucose', 'heart_disease']]
    
    # Handle zeros which are effectively missing values in this dataset (e.g. BP=0)
    cols_to_fix = ['bmi', 'blood_pressure', 'glucose']
    for col in cols_to_fix:
        df_clean[col] = df_clean[col].replace(0, df_clean[col].mean())

    # Generate Synthetic Cost Target (since it's not in the dataset)
    # Cost correlated with Age and Disease Status
    n_samples = len(df_clean)
    noise = np.random.normal(0, 500, n_samples)
    
    df_clean['hospital_cost'] = (
        1000 + 
        (df_clean['age'] * 50) + 
        (df_clean['heart_disease'] * 5000) + 
        noise
    )
    
    return df_clean

def train_models():
    print("Generating data...")
    df = generate_synthetic_data()
    df.to_csv("data/healthcare_data.csv", index=False)
    
    X = df[['age', 'bmi', 'blood_pressure', 'glucose']]
    y_class = df['heart_disease']
    y_reg = df['hospital_cost']
    
    # Advanced Model Training with Pipelines (Handling Scaling Automatically)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, mean_absolute_error, silhouette_score

    # 1. Classification Model (Gradient Boosting is often more accurate than RF)
    print("Training Enhanced Classification Model (Gradient Boosting)...")
    clf_pipeline = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42))
    clf_pipeline.fit(X, y_class)
    
    # Evaluate
    acc = accuracy_score(y_class, clf_pipeline.predict(X))
    print(f"Classification Accuracy (Training): {acc:.4f}")
    
    joblib.dump(clf_pipeline, "models/heart_disease_model.pkl")
    
    # 2. Regression Model
    print("Training Enhanced Regression Model...")
    reg_pipeline = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, random_state=42))
    reg_pipeline.fit(X, y_reg)
    
    # Evaluate
    mae = mean_absolute_error(y_reg, reg_pipeline.predict(X))
    print(f"Regression MAE: ${mae:.2f}")
    
    joblib.dump(reg_pipeline, "models/cost_prediction_model.pkl")
    
    # 3. Clustering (Unsupervised) - Scaling is CRITICAL for K-Means
    print("Training Scaled Clustering Model...")
    cluster_pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42))
    cluster_pipeline.fit(X)
    
    score = silhouette_score(X, cluster_pipeline.predict(X))
    print(f"Clustering Silhouette Score: {score:.4f}")
    
    joblib.dump(cluster_pipeline, "models/patient_cluster_model.pkl")
    
    print("Models saved in 'models/' directory.")

if __name__ == "__main__":
    train_models()
