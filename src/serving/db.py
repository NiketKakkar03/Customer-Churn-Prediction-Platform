import os
from typing import Generator

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_conn() -> Generator:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname=os.getenv("DB_NAME", "churn"),
        user=os.getenv("DB_USER", "churn_user"),
        password=os.getenv("DB_PASSWORD", "churn_pass"),
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_at_risk(conn, limit: int) -> tuple[list[dict], str | None]:
    """
    Return (customers, scoring_date_str) for the most recent scoring date.
    Queries the latest available date rather than today, so the API works
    correctly even if the nightly job ran slightly late.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT customer_id, churn_probability, rank, shap_reasons, scoring_date
            FROM churn_predictions
            WHERE scoring_date = (SELECT MAX(scoring_date) FROM churn_predictions)
            ORDER BY rank
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()

    if not rows:
        return [], None

    scoring_date = str(rows[0][4])
    customers = [
        {
            "customer_id": row[0],
            "churn_probability": float(row[1]),
            "rank": row[2],
            "reasons": row[3],  # psycopg2 deserialises JSONB to a Python list automatically
        }
        for row in rows
    ]
    return customers, scoring_date
