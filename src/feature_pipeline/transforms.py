import pandas as pd
import numpy as np


_SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

_YES_NO_COLS = {
    "PaperlessBilling": "paperless_billing",
    "OnlineSecurity": "has_online_security",
    "TechSupport": "has_tech_support",
    "Partner": "has_partner",
    "Dependents": "has_dependents",
}


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw IBM Telco rows into model-ready trajectory features.
    Returns one row per customer with all feature columns.
    """
    df = raw.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    df["tenure_months"] = df["tenure"].astype(int)
    df["monthly_charges"] = df["MonthlyCharges"].astype(float)
    df["total_charges"] = df["TotalCharges"].astype(float)

    # Trajectory proxy: current monthly charges vs lifetime average.
    # > 1.0 means the customer is paying more now than historically (e.g. after an upsell).
    avg_monthly = df["total_charges"] / df["tenure_months"].replace(0, np.nan)
    df["monthly_vs_avg_ratio"] = (df["monthly_charges"] / avg_monthly).round(4)

    df["contract_type"] = df["Contract"]
    df["payment_method"] = df["PaymentMethod"]
    df["internet_service"] = df["InternetService"]
    df["is_senior_citizen"] = df["SeniorCitizen"].astype(bool)

    for raw_col, feature_col in _YES_NO_COLS.items():
        df[feature_col] = df[raw_col].str.strip().str.lower() == "yes"

    df["num_products"] = df[_SERVICE_COLS].apply(
        lambda row: (row.str.strip().str.lower() == "yes").sum(), axis=1
    ).astype(int)

    feature_cols = [
        "customerID",
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
    result = df[feature_cols].rename(columns={"customerID": "customer_id"})
    return result.reset_index(drop=True)
