from prefect import flow, task
import subprocess
import sys
import os

@task
def run_training_script():
    print("Starting Model Training...")
    # Call the script we created in ml/train_models.py
    # using current python interpreter
    script_path = os.path.join("ml", "train_models.py")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Training failed: {result.stderr}")
        raise Exception("Training Script Failed")
    print(f"Training Output: {result.stdout}")
    return "Success"

@task
def validate_models():
    print("Validating models existence...")
    required_models = [
        "models/heart_disease_model.pkl", 
        "models/cost_prediction_model.pkl",
        "models/patient_cluster_model.pkl"
    ]
    missing = [m for m in required_models if not os.path.exists(m)]
    if missing:
        raise Exception(f"Missing models: {missing}")
    print("All models validated.")

@flow(name="Healthcare ML Training Pipeline")
def training_pipeline():
    status = run_training_script()
    validate_models()

if __name__ == "__main__":
    training_pipeline()
