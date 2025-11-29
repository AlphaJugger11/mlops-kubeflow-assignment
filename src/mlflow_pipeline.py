# src/mlflow_pipeline.py

import mlflow
import os
from pipeline_components import extract_data, preprocess_data, train_model, evaluate_model

# Paths
data_raw_path = "data/raw_data.csv"
train_path = "data/train.csv"
test_path = "data/test.csv"
model_path = "models/random_forest_model.pkl"
metrics_path = "metrics/eval_metrics.csv"

# Make sure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Start MLflow run
with mlflow.start_run(run_name="ml_pipeline_run"):
    
    # Step 1: Extract data
    extract_data(output_path=data_raw_path)
    
    # Step 2: Preprocess data
    preprocess_data(input_path=data_raw_path, train_path=train_path, test_path=test_path)
    
    # Step 3: Train model
    train_model(train_path=train_path, model_path=model_path)
    
    # Step 4: Evaluate model
    evaluate_model(test_path=test_path, model_path=model_path, metrics_path=metrics_path)

    print("Pipeline finished successfully! Check MLflow UI for run details.")
