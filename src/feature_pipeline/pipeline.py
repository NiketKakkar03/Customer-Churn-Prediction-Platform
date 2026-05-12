import datetime
import os
import pathlib

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from .transforms import compute_features
from .db import upsert_features

load_dotenv()

RAW_DATA_PATH = pathlib.Path(__file__).parents[2] / "data" / "raw" / "telco_churn.parquet"


def run(as_of_date: datetime.date | None = None) -> None:
    if as_of_date is None:
        as_of_date = datetime.date.today()

    raw = pd.read_parquet(RAW_DATA_PATH)
    features = compute_features(raw)

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname=os.getenv("DB_NAME", "churn"),
        user=os.getenv("DB_USER", "churn_user"),
        password=os.getenv("DB_PASSWORD", "churn_pass"),
    )
    try:
        upsert_features(conn, features, as_of_date)
    finally:
        conn.close()

    print(f"feature_pipeline: wrote {len(features)} rows for {as_of_date}")


if __name__ == "__main__":
    run()
