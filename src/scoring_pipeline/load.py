import datetime
import os
import pathlib

import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

CHAMPION_PATH = pathlib.Path(__file__).parents[2] / "models" / "champion.joblib"

_FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges", "monthly_vs_avg_ratio",
    "contract_type", "payment_method", "paperless_billing", "internet_service",
    "has_online_security", "has_tech_support", "num_products",
    "is_senior_citizen", "has_partner", "has_dependents",
]


def _engine():
    return create_engine(
        f"postgresql+psycopg2://"
        f"{os.getenv('DB_USER', 'churn_user')}:{os.getenv('DB_PASSWORD', 'churn_pass')}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5433')}"
        f"/{os.getenv('DB_NAME', 'churn')}"
    )


def load_champion():
    if not CHAMPION_PATH.exists():
        raise FileNotFoundError(
            f"No champion model at {CHAMPION_PATH}. Run model_trainer first."
        )
    return joblib.load(CHAMPION_PATH)


def load_scoring_features(as_of_date: datetime.date) -> pd.DataFrame:
    """
    Load all customers' features for scoring. Returns customer_id + feature columns.
    """
    query = text("""
        SELECT customer_id, {cols}
        FROM customer_features
        WHERE as_of_date = :as_of_date
    """.format(cols=", ".join(_FEATURE_COLS)))

    with _engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"as_of_date": as_of_date})
    return df
