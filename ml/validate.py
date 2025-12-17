import pandas as pd
import numpy as np
import joblib
# In a real scenario, we would use: from deepchecks.tabular import Dataset
# from deepchecks.tabular.checks import TrainTestFeatureDrift, ModelErrorAnalysis

def validate_data_integrity(df: pd.DataFrame):
    print("Running Data Integrity Checks...")
    # 1. Check for nulls
    if df.isnull().sum().sum() > 0:
        print("WARNING: Null values found!")
    else:
        print("PASSED: No null values.")

    # 2. Check for negative ages or invalid values
    if (df['age'] < 0).any():
        print("FAILED: Negative Age detected.")
    else:
        print("PASSED: Age values valid.")

def check_drift(train_df, new_df):
    print("\nChecking for Data Drift...")
    # Simple statistical drift check (Mean comparison)
    for col in ['age', 'bmi', 'blood_pressure', 'glucose']:
        mean_diff = abs(train_df[col].mean() - new_df[col].mean())
        if mean_diff > 5: # Arbitrary threshold
            print(f"DRIFT ALERT: {col} mean changed by {mean_diff:.2f}")
        else:
            print(f"PASSED: {col} drift within limits.")

if __name__ == "__main__":
    # Load training data
    try:
        train_df = pd.read_csv("data/healthcare_data.csv")
        validate_data_integrity(train_df)
        
        # Simulate new production data
        new_data = train_df.copy()
        new_data['age'] = new_data['age'] + 10 # Introduce drift
        
        check_drift(train_df, new_data)
        
    except FileNotFoundError:
        print("Data file not found. Run training first.")
