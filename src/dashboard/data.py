import os

import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@st.cache_data(ttl=300)
def load_at_risk(limit: int) -> tuple[list[dict], str | None]:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname=os.getenv("DB_NAME", "churn"),
        user=os.getenv("DB_USER", "churn_user"),
        password=os.getenv("DB_PASSWORD", "churn_pass"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT customer_id, churn_probability, rank, shap_reasons, scoring_date
                FROM churn_predictions
                WHERE scoring_date = (SELECT MAX(scoring_date) FROM churn_predictions)
                ORDER BY rank
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return [], None

    scoring_date = str(rows[0][4])
    customers = [
        {
            "rank": row[2],
            "customer_id": row[0],
            "churn_probability": float(row[1]),
            "reasons": ", ".join(row[3]) if row[3] else "—",
        }
        for row in rows
    ]
    return customers, scoring_date
