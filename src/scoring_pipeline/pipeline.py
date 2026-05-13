import datetime
import os

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from .load import load_champion, load_scoring_features
from .explain import compute_shap_values, top_reasons
from .db import write_predictions

load_dotenv()

_RAW_FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges", "monthly_vs_avg_ratio",
    "contract_type", "payment_method", "paperless_billing", "internet_service",
    "has_online_security", "has_tech_support", "num_products",
    "is_senior_citizen", "has_partner", "has_dependents",
]


def rank_customers(
    pipeline, features: pd.DataFrame, top_n: int | None = None
) -> pd.DataFrame:
    """
    Score features, attach SHAP reasons, return a ranked DataFrame.
    Pure function (no DB) — easy to unit test.
    """
    X = features[_RAW_FEATURE_COLS]
    probs = pipeline.predict_proba(X)[:, 1]

    shap_values, feature_names = compute_shap_values(pipeline, X)
    reasons = [
        top_reasons(shap_values[i], feature_names, X.iloc[i], k=3)
        for i in range(len(X))
    ]

    ranked = pd.DataFrame({
        "customer_id": features["customer_id"].values,
        "churn_probability": probs,
        "shap_reasons": reasons,
    }).sort_values("churn_probability", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    if top_n is not None:
        ranked = ranked.head(top_n)
    return ranked


def score(as_of_date: datetime.date | None = None, top_n: int | None = None) -> None:
    if as_of_date is None:
        as_of_date = datetime.date.today()

    pipeline = load_champion()
    features = load_scoring_features(as_of_date)

    if features.empty:
        print(f"scoring_pipeline: no features found for {as_of_date}")
        return

    ranked = rank_customers(pipeline, features, top_n=top_n)

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname=os.getenv("DB_NAME", "churn"),
        user=os.getenv("DB_USER", "churn_user"),
        password=os.getenv("DB_PASSWORD", "churn_pass"),
    )
    try:
        write_predictions(conn, ranked, as_of_date)
    finally:
        conn.close()

    print(f"scoring_pipeline: wrote {len(ranked)} ranked rows for {as_of_date}")


if __name__ == "__main__":
    score()
