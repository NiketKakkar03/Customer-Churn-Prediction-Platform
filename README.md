# Customer Churn Prediction Platform

An end-to-end ML system that scores every customer nightly for 30-day churn risk, explains predictions with SHAP, and surfaces the top-N at-risk customers to retention teams via a Streamlit dashboard and FastAPI endpoint.

## Architecture

```
IBM Telco Dataset → Feature Pipeline → Postgres
                                           │
                              Model Trainer (XGBoost / LR)
                                           │
                              Scoring Pipeline → Postgres
                                           │
                    ┌──────────────────────┴──────────────────────┐
                FastAPI /customers/at-risk          Streamlit Dashboard
                    │
              Prometheus ← Pushgateway ← Orchestrator (nightly)
                    │
                 Grafana (MLOps dashboard)
```

## Quick start — full stack

**Prerequisites:** Docker Desktop, `data/raw/telco_churn.parquet` present (see Data Setup below)

```bash
# Bring up Postgres, FastAPI, Streamlit, Prometheus, Grafana, and Pushgateway
docker compose up -d

# Run the nightly pipeline once to populate predictions
docker compose run --rm orchestrator
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| Grafana (admin / admin) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Pushgateway | http://localhost:9091 |

## Data setup

Download the IBM Telco Customer Churn dataset and convert it to parquet:

```bash
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p /tmp
unzip /tmp/telco-customer-churn.zip -d /tmp
python -c "
import pandas as pd
df = pd.read_csv('/tmp/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.to_parquet('data/raw/telco_churn.parquet', index=False)
print(f'Ready: {len(df)} rows')
"
```

## Local development (without Docker)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# Start Postgres only
docker compose up -d postgres

# Run the full nightly pipeline
python -m src.orchestrator.run

# Serve the API
uvicorn src.serving.app:app --reload

# Run the dashboard
streamlit run src/dashboard/app.py
```

## Running the nightly orchestrator

```bash
# In Docker (one-shot, then container exits)
docker compose run --rm orchestrator

# Locally
python -m src.orchestrator.run
```

The orchestrator runs four steps in sequence:
1. **Feature pipeline** — computes features from raw data and upserts into Postgres
2. **Scoring pipeline** — scores all customers, ranks by churn probability, writes to Postgres
3. **Drift monitor** — runs Evidently DataDriftPreset against the training baseline; writes `models/drift_report.json`
4. **Retraining** (conditional) — if drift is detected and the 7-day cooldown has passed, retrains and promotes a new champion model

## Running tests

```bash
# Unit tests (no Postgres required)
pytest

# Integration tests (requires Postgres running on port 5433)
pytest -m integration
```

## CI / CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Pull request to `main` | Runs 50 unit tests + builds all 3 Docker images |
| `promote.yml` | Push to `main` | Downloads data, runs feature pipeline + quality gate, commits promoted model artifacts |

The quality gate (`scripts/quality_gate.py`) trains a candidate model and promotes it only if its Precision@50 meets or beats the current champion stored in `models/champion_meta.json`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | Postgres host |
| `DB_PORT` | `5433` | Postgres port (5432 inside Docker Compose) |
| `DB_NAME` | `churn` | Database name |
| `DB_USER` | `churn_user` | Database user |
| `DB_PASSWORD` | `churn_pass` | Database password |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `PUSHGATEWAY_URL` | `localhost:9091` | Prometheus Pushgateway for batch metrics |
