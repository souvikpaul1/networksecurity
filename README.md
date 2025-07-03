

---


# 🔐 Network Security Threat Detection — End-to-End MLOps Project

This repository showcases a **real-world, production-grade MLOps pipeline** built for detecting threats in network traffic. The project integrates the entire ML lifecycle—from **data ingestion and preprocessing** to **model training**, **cloud-based deployment**, and **CI/CD automation**—leveraging both **ML** and **DevOps best practices**.

Designed to highlight your ability to build scalable, maintainable, and cloud-native ML systems, this project is perfect for demonstrating practical expertise in deploying AI solutions using **FastAPI**, **AWS**, **Docker**, **GitHub Actions**, and more.

---

## 🚀 Project Highlights

| ✅ Feature | 💻 Tools / Services |
|-----------|---------------------|
| Environment Setup | `conda`, `requirements.txt`, `setup.py` |
| Modular Project Structure | Python Package with `__init__.py` |
| Data Ingestion Sources | `MongoDB Atlas`, `S3`, `Public APIs`, `CSV` |
| ETL Pipeline | Extract → Transform → Load |
| Data Validation | Schema checks, drift detection |
| Data Transformation | `KNN Imputer`, normalization |
| Model Training | `scikit-learn`, `MLflow`, `DagsHub` |
| Experiment Tracking | `MLflow`, `DagsHub` |
| Model Serving | `FastAPI`, `Uvicorn`, Swagger UI |
| CI/CD Automation | `GitHub Actions`, `Docker`, `AWS ECR`, `EC2` |
| Deployment & Registry | `Docker`, `AWS S3`, `EC2 self-hosted runner` |

---

````markdown
## 📁 Project Structure

```text
networksecurity/
│
├── cloud/             # AWS, MongoDB, S3 interactions
├── components/        # Pipeline components (training, validation, etc.)
├── constants/         # Config and constant values
├── entity/            # Custom data models and schemas
├── exception/         # Custom exception handling
├── logging/           # Logging utility
├── pipeline/          # End-to-end ML pipelines
├── utils/             # Helper functions and reusable logic
│
├── .github/workflows/main.yaml   # CI/CD workflow for GitHub Actions
├── notebooks/                    # EDA and experimentation
├── Network_data/                 # Input and processed datasets
├── .env                          # Secure environment variables
├── .gitignore
├── Dockerfile                    # Docker configuration
├── project_workflow.txt          # Step-by-step execution plan
├── README.md
├── requirements.txt
├── setup.py                      # Python package setup
````

Each module under `networksecurity/` includes an `__init__.py` for clean packaging and import handling.

---

## ⚙️ Environment Setup

```bash
conda create -n network_security python=3.10
conda activate network_security
pip install -r requirements.txt
```

> 🛠️ The entire project is structured as a reusable Python package (`setup.py`), ensuring reproducibility and smooth deployment across environments.

```

```


## 🔄 ETL Pipeline (Extract, Transform, Load)

| Stage         | Details                                                      |
| ------------- | ------------------------------------------------------------ |
| **Extract**   | Data pulled from `APIs`, `S3 Bucket`, `MongoDB`, `Paid APIs` |
| **Transform** | JSON processing, basic cleaning, KNN Imputer                 |
| **Load**      | Pushed to `MongoDB Atlas`, `AWS DynamoDB`, `SQL`, `S3`       |

### MongoDB Details

* Database: `network_security`
* Collection: `network_data`
* Ingestion script: `push_data.py`
* Records Pushed: **11,055**

.env file securely stores connection strings (not pushed to GitHub)

---

## 📥 Data Ingestion Module

Key Features:

* Export MongoDB data to Pandas DataFrame
* Save feature store to avoid repeated DB queries

Artifacts:

* `ingestion_artifact.pkl`
* Feature store cache for optimized pipeline runs

---

## ✅ Data Validation

* Checks schema consistency
* Validates data drift between train/test sets using statistical thresholds
* Ensures training integrity

---

## 🔄 Data Transformation

* Implements **KNN Imputer** for missing values
* Applies feature engineering on top of ingestion artifacts

---

## 🧠 Model Training

| Feature        | Description                                    |
| -------------- | ---------------------------------------------- |
| **MLflow**     | Tracks experiments, metrics, artifacts         |
| **Dagshub**    | Remote Git repo + MLflow tracking              |
| **Algorithms** | Logistic Regression, Tree-based models         |
| **FastAPI**    | Exposes trained model as a prediction API      |
| **Swagger UI** | Interactive test interface for model endpoints |

> Note: Resolved common connection issues with Dagshub during model training.

---

## ☁️ Cloud Integration

### ✅ AWS S3

* Stores final models and pipeline artifacts
* Bucket versioning enabled

### ✅ IAM & AWS CLI

* Configured using a secure IAM user for AWS SDK
* S3 bucket operations via Python SDK

---

## 🐳 Docker & CI/CD

### Docker Image

```bash
docker build -t network-security-app .
docker run -p 5000:5000 network-security-app
```

### GitHub Actions Workflow (`main.yaml`)

* Triggers on push/merge to `main`
* Builds Docker image
* Pushes to AWS ECR

#### GitHub Secrets Used:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
AWS_ECR_LOGIN_URI
```

---

## 🚀 Final Deployment (EC2 Self-Hosted Runner)

> 💡 **Why EC2 over EKS?** MVPs and low-cost deployments are ideal for EC2 runners.

| Feature          | EC2 Self-Hosted | EKS (Kubernetes) |
| ---------------- | --------------- | ---------------- |
| 💸 Cost          | ✅ Low           | ❌ High           |
| ⚙️ Complexity    | ✅ Simple        | ❌ Complex        |
| 🔁 Auto Recovery | ❌ Manual        | ✅ Built-in       |
| 🔗 Use Case      | MVP, Demos      | Production Scale |

### Steps:

1. Create EC2 (t2.medium)
2. Setup Docker, connect EC2 to GitHub as self-hosted runner
3. Enable port access on **port 5000**
4. Final run: `./run.sh` on EC2 terminal

---

## 🔁 Workflow Diagram

```text
+------------------+
|  MongoDB / CSV   |
+--------+---------+
         |
         v
+------------------+
| Data Ingestion   |
+--------+---------+
         |
         v
+------------------+
| Data Validation  |
+--------+---------+
         |
         v
+------------------+
| Data Transform   |
+--------+---------+
         |
         v
+------------------+
| Model Training   |
+--------+---------+
         |
         v
+------------------+
|  MLflow & DVC    |
+--------+---------+
         |
         v
+------------------+
| Docker Container |
+--------+---------+
         |
         v
+--------------------------+
|  AWS ECR + EC2 Runner    |
+--------------------------+
```

---

## 📌 Summary

| Task                     | Status                 |
| ------------------------ | ---------------------- |
| Environment & Structure  | ✅ Done                 |
| MongoDB Ingestion        | ✅ 11,055 Records       |
| ETL + Drift Validation   | ✅ Completed            |
| Model Training & Logging | ✅ via MLflow + Dagshub |
| Docker + CI/CD           | ✅ Pushed to ECR        |
| Self-Hosted Runner       | ✅ Deployed on EC2      |
| Model Serving            | ✅ Live on port 5000    |

---
