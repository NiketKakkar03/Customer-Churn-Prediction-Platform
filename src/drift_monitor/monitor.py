import datetime
import json
import os
import pathlib
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import create_engine, text

load_dotenv()

MODELS_DIR = pathlib.Path(__file__).parents[2] / "models"
BASELINE_PATH = MODELS_DIR / "training_baseline.parquet"
DRIFT_REPORT_PATH = MODELS_DIR / "drift_report.json"
CHAMPION_META_PATH = MODELS_DIR / "champion_meta.json"

_COOLDOWN_DAYS = 7
_DRIFT_SHARE_THRESHOLD = 0.5  # retrain when >50% of features drift


@dataclass
class DriftResult:
    drift_detected: bool
    should_retrain: bool
    share_drifted_columns: float
    checked_at: str
    details: dict = field(default_factory=dict)


def check(as_of_date: datetime.date | None = None) -> DriftResult:
    if as_of_date is None:
        as_of_date = datetime.date.today()

    reference = _load_baseline()
    current = _load_current_features(as_of_date)

    # drift_share=0.3: flag dataset drift when >30% of features shift.
    # Default (0.5) is too conservative for our mix of numeric + boolean + categorical columns.
    report = Report(metrics=[DataDriftPreset(drift_share=0.3)])
    report.run(reference_data=reference, current_data=current)
    result_dict = report.as_dict()

    drift_result = result_dict["metrics"][0]["result"]
    drift_detected = drift_result["dataset_drift"]
    share_drifted = drift_result["share_of_drifted_columns"]

    should_retrain = drift_detected and _cooldown_passed()

    dr = DriftResult(
        drift_detected=drift_detected,
        should_retrain=should_retrain,
        share_drifted_columns=share_drifted,
        checked_at=str(as_of_date),
        details={col: info for col, info in drift_result.get("drift_by_columns", {}).items()},
    )
    _save_report(dr)
    return dr


def _load_baseline() -> pd.DataFrame:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"No training baseline at {BASELINE_PATH}. Run model_trainer first."
        )
    return pd.read_parquet(BASELINE_PATH)


def _load_current_features(as_of_date: datetime.date) -> pd.DataFrame:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    dbname = os.getenv("DB_NAME", "churn")
    user = os.getenv("DB_USER", "churn_user")
    password = os.getenv("DB_PASSWORD", "churn_pass")
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")

    query = text("""
        SELECT tenure_months, monthly_charges, total_charges, monthly_vs_avg_ratio,
               num_products, paperless_billing, has_online_security, has_tech_support,
               is_senior_citizen, has_partner, has_dependents,
               contract_type, payment_method, internet_service
        FROM customer_features
        WHERE as_of_date = :as_of_date
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"as_of_date": as_of_date})


def _cooldown_passed() -> bool:
    if not CHAMPION_META_PATH.exists():
        return True
    meta = json.loads(CHAMPION_META_PATH.read_text())
    trained_on = datetime.date.fromisoformat(meta.get("trained_on", "2000-01-01"))
    return (datetime.date.today() - trained_on).days >= _COOLDOWN_DAYS


def _save_report(dr: DriftResult) -> None:
    DRIFT_REPORT_PATH.write_text(json.dumps({
        "drift_detected": dr.drift_detected,
        "should_retrain": dr.should_retrain,
        "share_drifted_columns": dr.share_drifted_columns,
        "checked_at": dr.checked_at,
    }, indent=2))
