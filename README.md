# 🚀 MLOps Assignment 4 — Kubeflow Pipelines | DVC | MLflow | CI/CD | Jenkins | GitHub Actions

This repository contains the complete implementation of **Assignment 4 for MLOps**, integrating modern MLOps practices such as **experiment tracking**, **data versioning**, **model registry**, **CI/CD automation**, and **pipeline orchestration** using **Kubeflow Pipelines**.

It ensures **reproducibility, scalability, automation, and collaboration** across the entire ML lifecycle.

---

## 📌 Project Overview

This project demonstrates a fully operational MLOps workflow consisting of:

✔ **Data Versioning with DVC** (remote storage: AWS S3)
✔ **Experiment Tracking with MLflow**
✔ **Model Training Pipeline** (train → evaluate → register)
✔ **Containerization (Dockerfile)**
✔ **Automated CI (GitHub Actions) & CI/CD (Jenkins pipeline)**
✔ **Pipeline Orchestration using Kubeflow Pipelines**
✔ **Metrics logging & artifact tracking**
✔ **Reproducible directory structure**

The ML task focuses on **training a regression model** using the provided `train.csv` and `test.csv` datasets.

---

## 📂 Repository Structure

```
MLOpsKMFlow/
│── components/              # Kubeflow components
│── data/                    # Raw/train/test data
│── media/                   # Screenshots for assignment proof
│── metrics/                 # Metrics logged with DVC
│── mlruns/                  # MLflow experiment tracking
│── models/                  # Model registry
│── src/                     # Source code
│── .github/workflows/       # GitHub Actions CI pipeline
│── .dvc/                    # DVC internal structure
│── Dockerfile               # Containerization
│── Jenkinsfile              # Jenkins CI/CD pipeline
│── pipeline.py              # Kubeflow Pipeline
│── requirements.txt         # Dependencies
│── data.dvc                 # Data tracking with DVC
│── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ☁️ 2️⃣ Configure DVC with S3

### Initialize DVC

```bash
dvc init
```

### Add S3 Remote

```bash
dvc remote add -d myremote s3://your-bucket/path
```

### Push Data

```bash
dvc push
```

---

## 📊 3️⃣ MLflow Tracking

Start MLflow UI:

```bash
mlflow ui --port 5000
```

Logs include:

* metrics (MAE, MSE, R²)
* parameters
* artifacts
* models

---

## 🐳 4️⃣ Docker Setup

### Build

```bash
docker build -t mlops-kmflow .
```

### Run

```bash
docker run -it mlops-kmflow
```

---

## 🤖 5️⃣ Kubeflow Pipeline

Pipeline includes:

* load_data
* train_model
* evaluate_model
* register_model

### Compile Pipeline

```bash
python pipeline.py
```

Upload generated YAML to Kubeflow UI.

---

## 🧪 6️⃣ GitHub Actions CI

Located in:

```
.github/workflows/ci.yml
```

Performs:

* Install dependencies
* Linting
* DVC pull
* Pipeline test run

---

## 🔁 7️⃣ Jenkins CI/CD

The Jenkinsfile automates:

* Git checkout
* Python setup
* Model training
* Metrics export

Screenshot located in `/media`.

---

## 📜 Pipeline Walkthrough

### Prepare environment

```bash
dvc pull
pip install -r requirements.txt
```

### Run training

```bash
python src/train.py
```

### Evaluate model

```bash
python src/evaluate.py
```

### Compile Kubeflow Pipeline

```bash
python pipeline.py
```

---

## 🖼️ Screenshots for Assignment

All screenshots are in `/media`:

* fileStructure.png
* dvcInS3.png
* dvcPushStatus.png
* mlflowUI.png
* task2mod1.png
* task2mlflowRuns.png
* task3artifacts.png
* task4CiYamlcontents.png
* task4githubActionsRunning.png
* task4jenkinsFileContents.png
* modelMetrics.png

---

## 🎯 Conclusion

This repository demonstrates a complete **end-to-end MLOps pipeline** featuring:

✔ Automated & versioned workflows
✔ Reproducible ML experiments
✔ Cloud-backed data versioning
✔ Model tracking + CI/CD automation

If you need polishing or want badges added—just tell me!
