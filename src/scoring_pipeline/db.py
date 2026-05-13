import datetime
import json

import pandas as pd
import psycopg2.extras


def write_predictions(conn, ranked: pd.DataFrame, scoring_date: datetime.date) -> None:
    """
    Write a ranked predictions DataFrame to churn_predictions.
    Idempotent: clears all rows for scoring_date first, then inserts fresh ranking.
    `ranked` must have columns: customer_id, churn_probability, rank, shap_reasons (list[str]).
    """
    records = [
        (
            row.customer_id,
            scoring_date,
            float(row.churn_probability),
            int(row.rank),
            json.dumps(list(row.shap_reasons)),
        )
        for row in ranked.itertuples(index=False)
    ]

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM churn_predictions WHERE scoring_date = %s",
            (scoring_date,),
        )
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO churn_predictions
                (customer_id, scoring_date, churn_probability, rank, shap_reasons)
            VALUES %s
            """,
            records,
        )
    conn.commit()
