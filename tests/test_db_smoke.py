import os
import psycopg2
import pytest

from dotenv import load_dotenv
load_dotenv()

@pytest.fixture
def db_conn():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname=os.getenv("DB_NAME", "churn"),
        user=os.getenv("DB_USER", "churn_user"),
        password=os.getenv("DB_PASSWORD", "churn_pass"),
    )
    yield conn
    conn.close()


def test_db_connection(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1


def test_customer_features_table_exists(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'customer_features'
            )
        """)
        assert cur.fetchone()[0] is True


def test_churn_predictions_table_exists(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'churn_predictions'
            )
        """)
        assert cur.fetchone()[0] is True
