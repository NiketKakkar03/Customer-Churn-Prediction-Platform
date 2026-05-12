The Project: Customer Churn Prediction Platform with MLOps
This one project hits almost every major bullet in the BMO JD simultaneously. Here's the full scope:

What You Build
A complete, end-to-end ML system — not just a model, but a platform:

Data Pipeline — Ingest and clean a telecom/banking churn dataset (IBM Telco or similar), engineer features (recency, frequency, engagement metrics), store them in a lightweight feature store (Postgres or Redis)

Model Training — Train XGBoost + a Logistic Regression baseline, log all experiments with MLflow, use time-aware cross-validation (no shuffling)

Explainability Layer — Add SHAP values so every prediction comes with a reason ("this customer is likely to churn because their support calls increased 3x") — critical for a risk-aware bank

REST API Serving — Wrap the model in a FastAPI endpoint, containerize with Docker, add Prometheus metrics (latency, prediction distribution, request volume)

Drift Detection + Auto-Retraining — Monitor incoming feature distributions against training baseline using Evidently AI; trigger retraining via a scheduled job when drift is detected

CI/CD Pipeline — GitHub Actions workflow that runs tests, validates model performance on a holdout set, and only promotes the new model if it beats the current one

Monitoring Dashboard — A simple Grafana or Streamlit dashboard showing live prediction stats, drift alerts, and model performance over time


Tech Stack
Python, XGBoost, scikit-learn — core ML

MLflow — experiment tracking

FastAPI + Docker — model serving

Evidently AI — drift detection

Prometheus + Grafana or Streamlit — monitoring

GitHub Actions — CI/CD

PostgreSQL — feature store