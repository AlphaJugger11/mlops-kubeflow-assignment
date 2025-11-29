# src/pipeline_components.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import mlflow
import mlflow.sklearn
import dvc.api

# ----------------------------
# 1. Data Extraction Component
# ----------------------------
def extract_data(output_path: str):
    """Fetch dataset from DVC remote storage."""
    with dvc.api.open('data/raw_data.csv', repo='.', mode='rb') as f:
        df = pd.read_csv(f)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    mlflow.log_artifact(output_path)

# ----------------------------
# 2. Data Preprocessing Component
# ----------------------------
def preprocess_data(input_path: str, train_path: str, test_path: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(input_path)
    df = df.fillna(df.mean())

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Preprocessed train/test saved to {train_path} and {test_path}")

    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

# ----------------------------
# 3. Model Training Component
# ----------------------------
def train_model(train_path: str, model_path: str):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor  # it's a regression problem
    import joblib

    train_df = pd.read_csv(train_path)

    # Label column
    label_column = 'MedHouseVal'

    X = train_df.drop(label_column, axis=1)
    y = train_df[label_column]

    # Using RandomForestRegressor because MedHouseVal is continuous
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# ----------------------------
# 4. Model Evaluation Component
# ----------------------------
def evaluate_model(test_path: str, model_path: str, metrics_path: str):
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score

    test_df = pd.read_csv(test_path)
    label_column = 'MedHouseVal'

    X_test = test_df.drop(label_column, axis=1)
    y_test = test_df[label_column]

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    metrics = {
        'MSE': mean_squared_error(y_test, preds),
        'R2': r2_score(y_test, preds)
    }

    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    print(f"Metrics saved to {metrics_path}")
