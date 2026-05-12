import datetime
import pandas as pd
import psycopg2
import psycopg2.extras


def upsert_features(conn, features: pd.DataFrame, as_of_date: datetime.date) -> None:
    records = [
        (
            row.customer_id,
            as_of_date,
            _int_or_none(row.tenure_months),
            _float_or_none(row.monthly_charges),
            _float_or_none(row.total_charges),
            _float_or_none(row.monthly_vs_avg_ratio),
            row.contract_type,
            row.payment_method,
            bool(row.paperless_billing),
            row.internet_service,
            bool(row.has_online_security),
            bool(row.has_tech_support),
            int(row.num_products),
            bool(row.is_senior_citizen),
            bool(row.has_partner),
            bool(row.has_dependents),
        )
        for row in features.itertuples(index=False)
    ]

    sql = """
        INSERT INTO customer_features (
            customer_id, as_of_date, tenure_months, monthly_charges, total_charges,
            monthly_vs_avg_ratio, contract_type, payment_method, paperless_billing,
            internet_service, has_online_security, has_tech_support, num_products,
            is_senior_citizen, has_partner, has_dependents
        ) VALUES %s
        ON CONFLICT (customer_id, as_of_date) DO UPDATE SET
            tenure_months        = EXCLUDED.tenure_months,
            monthly_charges      = EXCLUDED.monthly_charges,
            total_charges        = EXCLUDED.total_charges,
            monthly_vs_avg_ratio = EXCLUDED.monthly_vs_avg_ratio,
            contract_type        = EXCLUDED.contract_type,
            payment_method       = EXCLUDED.payment_method,
            paperless_billing    = EXCLUDED.paperless_billing,
            internet_service     = EXCLUDED.internet_service,
            has_online_security  = EXCLUDED.has_online_security,
            has_tech_support     = EXCLUDED.has_tech_support,
            num_products         = EXCLUDED.num_products,
            is_senior_citizen    = EXCLUDED.is_senior_citizen,
            has_partner          = EXCLUDED.has_partner,
            has_dependents       = EXCLUDED.has_dependents,
            created_at           = NOW()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, records)
    conn.commit()


def _int_or_none(v):
    return None if pd.isna(v) else int(v)


def _float_or_none(v):
    return None if pd.isna(v) else float(v)
