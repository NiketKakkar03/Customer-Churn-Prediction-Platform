import pandas as pd
import numpy as np
import pytest

from src.feature_pipeline.transforms import compute_features


def _make_raw(**overrides) -> pd.DataFrame:
    base = {
        "customerID": "C001",
        "tenure": 12,
        "MonthlyCharges": 60.0,
        "TotalCharges": "600.0",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "SeniorCitizen": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestMonthlyVsAvgRatio:
    def test_ratio_equals_one_when_charges_are_flat(self):
        # MonthlyCharges == TotalCharges / tenure → ratio should be 1.0
        df = compute_features(_make_raw(tenure=10, MonthlyCharges=50.0, TotalCharges="500.0"))
        assert df.loc[0, "monthly_vs_avg_ratio"] == pytest.approx(1.0, rel=1e-3)

    def test_ratio_greater_than_one_when_current_charges_higher(self):
        # Customer started cheap and was upsold — current monthly > historical avg
        df = compute_features(_make_raw(tenure=12, MonthlyCharges=80.0, TotalCharges="600.0"))
        assert df.loc[0, "monthly_vs_avg_ratio"] > 1.0

    def test_ratio_less_than_one_when_current_charges_lower(self):
        df = compute_features(_make_raw(tenure=12, MonthlyCharges=40.0, TotalCharges="600.0"))
        assert df.loc[0, "monthly_vs_avg_ratio"] < 1.0

    def test_zero_tenure_produces_null_ratio(self):
        df = compute_features(_make_raw(tenure=0, MonthlyCharges=50.0, TotalCharges="0.0"))
        assert pd.isna(df.loc[0, "monthly_vs_avg_ratio"])


class TestNumProducts:
    def test_counts_only_yes_values(self):
        df = compute_features(_make_raw(
            PhoneService="Yes", MultipleLines="Yes", OnlineSecurity="Yes",
            OnlineBackup="No", DeviceProtection="No", TechSupport="No",
            StreamingTV="No", StreamingMovies="No",
        ))
        assert df.loc[0, "num_products"] == 3

    def test_no_services_gives_zero(self):
        df = compute_features(_make_raw(
            PhoneService="No", MultipleLines="No", OnlineSecurity="No",
            OnlineBackup="No", DeviceProtection="No", TechSupport="No",
            StreamingTV="No", StreamingMovies="No",
        ))
        assert df.loc[0, "num_products"] == 0


class TestBooleanFlags:
    def test_paperless_billing_yes(self):
        df = compute_features(_make_raw(PaperlessBilling="Yes"))
        assert df.loc[0, "paperless_billing"] == True

    def test_paperless_billing_no(self):
        df = compute_features(_make_raw(PaperlessBilling="No"))
        assert df.loc[0, "paperless_billing"] == False

    def test_is_senior_citizen(self):
        df = compute_features(_make_raw(SeniorCitizen=1))
        assert df.loc[0, "is_senior_citizen"] == True


class TestBoundaryCases:
    def test_invalid_total_charges_row_is_dropped(self):
        raw = pd.concat([
            _make_raw(customerID="C001", TotalCharges="600.0"),
            _make_raw(customerID="C002", TotalCharges=" "),  # IBM Telco blank for new customers
        ])
        result = compute_features(raw)
        assert len(result) == 1
        assert result.loc[0, "customer_id"] == "C001"

    def test_multiple_customers_all_present(self):
        raw = pd.concat([
            _make_raw(customerID="C001"),
            _make_raw(customerID="C002"),
            _make_raw(customerID="C003"),
        ])
        result = compute_features(raw)
        assert len(result) == 3
        assert set(result["customer_id"]) == {"C001", "C002", "C003"}

    def test_output_columns_match_schema(self):
        expected = {
            "customer_id", "tenure_months", "monthly_charges", "total_charges",
            "monthly_vs_avg_ratio", "contract_type", "payment_method",
            "paperless_billing", "internet_service", "has_online_security",
            "has_tech_support", "num_products", "is_senior_citizen",
            "has_partner", "has_dependents",
        }
        df = compute_features(_make_raw())
        assert set(df.columns) == expected
