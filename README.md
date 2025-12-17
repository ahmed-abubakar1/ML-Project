# End-to-End Machine Learning System: Healthcare Analytics
## Project Title: Design and Deploy an End-to-End Machine Learning System

**Domain**: Healthcare (Disease Prediction, Cost Estimation, Patient Clustering)

## Project Overview
This project implements a full-stack ML Engineering system with:
- **FastAPI**: Serving real-time predictions and a **Web UI** for user interaction.
- **GitHub Actions**: CI/CD pipeline for automated testing and docker builds.
- **Prefect**: Orchestration of data ingestion, training, and evaluation pipelines.
- **Automated Validation**: Scripts to check data integrity and model drift.
- **Docker**: Containerization of the API and Services.

## Structure
- `app/`: FastAPI application (`main.py`) and Web UI (`static/`).
- `ml/`: Machine Learning scripts:
    - `train_models.py`: Generates synthetic data and trains 3 models.
    - `validate.py`: Checks for data integrity and drift.
- `prefect_flows/`: Prefect workflows (`pipeline.py`).
- `models/`: Serialized model files (.pkl).
- `tests/`: Automated API tests.
- `.github/workflows/`: CI/CD configuration.
- `Dockerfile` & `docker-compose.yml`: Container setup.

## Setup & Running

### 1. Initialize Environment
```bash
pip install -r requirements.txt
```

### 2. Train Models
Run the ML pipeline to generate data and models:
```bash
python ml/train_models.py
```

### 3. Run Validation
```bash
python ml/validate.py
```

### 4. Start API (Locally)
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` to test endpoints.

### 5. Orchestration (Prefect)
```bash
python prefect_flows/pipeline.py
```

### 6. Docker
```bash
docker-compose up --build
```
