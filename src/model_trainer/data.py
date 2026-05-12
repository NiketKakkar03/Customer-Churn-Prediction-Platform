import datetime
import os

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_FEATURE_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "monthly_vs_avg_ratio",
    "contract_type",
    "payment_method",
    "paperless_billing",
    "internet_service",
    "has_online_security",
    "has_tech_support",
    "num_products",
    "is_senior_citizen",
    "has_partner",
    "has_dependents",
]


def _engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    dbname = os.getenv("DB_NAME", "churn")
    user = os.getenv("DB_USER", "churn_user")
    password = os.getenv("DB_PASSWORD", "churn_pass")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")


def load_features(as_of_date: datetime.date) -> pd.DataFrame:
    query = text("""
        SELECT customer_id, {cols}
        FROM customer_features
        WHERE as_of_date = :as_of_date
    """.format(cols=", ".join(_FEATURE_COLS)))

    with _engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"as_of_date": as_of_date})

    labels = _load_labels()
    df = df.merge(labels, on="customer_id", how="inner")
    return df


def _load_labels() -> pd.DataFrame:
    import pathlib
    raw_path = pathlib.Path(__file__).parents[2] / "data" / "raw" / "telco_churn.parquet"
    raw = pd.read_parquet(raw_path, columns=["customerID", "Churn"])
    raw = raw.rename(columns={"customerID": "customer_id"})
    raw["churned"] = (raw["Churn"].str.strip().str.lower() == "yes").astype(int)
    return raw[["customer_id", "churned"]]
