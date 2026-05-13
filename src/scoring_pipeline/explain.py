from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def compute_shap_values(
    pipeline: Pipeline, raw_features: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for the positive (churn) class.
    Returns (shap_values matrix [n_customers x n_features], feature_names).
    """
    preprocessor = pipeline.named_steps["pre"]
    classifier = pipeline.named_steps["clf"]
    X_transformed = preprocessor.transform(raw_features)
    feature_names = list(preprocessor.get_feature_names_out())

    if isinstance(classifier, LogisticRegression):
        background = shap.sample(X_transformed, min(100, X_transformed.shape[0]), random_state=42)
        explainer = shap.LinearExplainer(classifier, background)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # XGBoost / tree-based — TreeExplainer is fast and exact
        explainer = shap.TreeExplainer(classifier)
        raw_shap = explainer.shap_values(X_transformed)
        # TreeExplainer for binary XGB returns single array (margin for class 1)
        shap_values = raw_shap if not isinstance(raw_shap, list) else raw_shap[1]

    return shap_values, feature_names


# Maps transformed feature names (with ColumnTransformer prefixes) → human-readable templates.
# Templates take the customer's raw feature value and return a sentence fragment.
def _format_reason(feature_name: str, raw_row: pd.Series) -> str | None:
    name = feature_name.split("__", 1)[-1]  # strip "num__", "cat__", "bool__" prefixes

    # Numeric features
    if name == "tenure_months":
        v = int(raw_row["tenure_months"])
        return f"short tenure ({v} months)" if v < 12 else f"tenure of {v} months"
    if name == "monthly_charges":
        return f"monthly charges of ${raw_row['monthly_charges']:.0f}"
    if name == "total_charges":
        return f"lifetime spend of ${raw_row['total_charges']:.0f}"
    if name == "monthly_vs_avg_ratio":
        v = float(raw_row["monthly_vs_avg_ratio"])
        if v > 1.10:
            return f"monthly charges {(v - 1) * 100:.0f}% above lifetime average"
        if v < 0.90:
            return f"monthly charges {(1 - v) * 100:.0f}% below lifetime average"
        return "monthly charges near lifetime average"
    if name == "num_products":
        v = int(raw_row["num_products"])
        return f"subscribed to {v} service{'s' if v != 1 else ''}"

    # Boolean features
    if name == "paperless_billing":
        return "uses paperless billing" if raw_row["paperless_billing"] else "uses paper billing"
    if name == "has_online_security":
        return "has online security" if raw_row["has_online_security"] else "no online security"
    if name == "has_tech_support":
        return "has tech support" if raw_row["has_tech_support"] else "no tech support"
    if name == "is_senior_citizen":
        return "senior citizen" if raw_row["is_senior_citizen"] else None
    if name == "has_partner":
        return "has a partner" if raw_row["has_partner"] else "no partner"
    if name == "has_dependents":
        return "has dependents" if raw_row["has_dependents"] else "no dependents"

    # One-hot encoded categoricals: "contract_type_Month-to-month", "payment_method_Mailed check", etc.
    # Match against known categorical columns explicitly (column names themselves contain underscores).
    for col in ("contract_type", "payment_method", "internet_service"):
        prefix = f"{col}_"
        if name.startswith(prefix):
            value = name[len(prefix):]
            if str(raw_row.get(col)) != value:
                return None  # this one-hot column is not active for this customer
            if col == "contract_type":
                return f"{value.lower()} contract"
            if col == "payment_method":
                return f"pays by {value.lower()}"
            if col == "internet_service":
                return (
                    "no internet service" if value.lower() == "no"
                    else f"{value.lower()} internet"
                )
    return None


def top_reasons(
    shap_row: np.ndarray, feature_names: list[str], raw_row: pd.Series, k: int = 3
) -> list[str]:
    """
    Return the top-k human-readable reasons this customer was flagged.
    Picks features with the largest *positive* SHAP contribution (drivers of churn risk).
    """
    # Sort descending by SHAP value (most positive first)
    order = np.argsort(shap_row)[::-1]
    reasons: list[str] = []
    for idx in order:
        if shap_row[idx] <= 0:
            break  # we've exhausted positive contributors
        text = _format_reason(feature_names[idx], raw_row)
        if text and text not in reasons:
            reasons.append(text)
        if len(reasons) >= k:
            break
    return reasons
