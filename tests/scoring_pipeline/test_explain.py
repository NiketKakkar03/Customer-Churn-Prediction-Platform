import numpy as np
import pandas as pd
import pytest

from src.scoring_pipeline.explain import top_reasons, _format_reason


@pytest.fixture
def raw_row():
    return pd.Series({
        "tenure_months": 3,
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "monthly_vs_avg_ratio": 1.20,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "paperless_billing": True,
        "internet_service": "Fiber optic",
        "has_online_security": False,
        "has_tech_support": False,
        "num_products": 2,
        "is_senior_citizen": False,
        "has_partner": False,
        "has_dependents": False,
    })


class TestFormatReason:
    def test_short_tenure_gets_short_qualifier(self, raw_row):
        assert "short tenure" in _format_reason("num__tenure_months", raw_row)

    def test_long_tenure_gets_neutral_phrasing(self, raw_row):
        raw_row["tenure_months"] = 48
        assert _format_reason("num__tenure_months", raw_row) == "tenure of 48 months"

    def test_ratio_above_threshold_describes_increase(self, raw_row):
        result = _format_reason("num__monthly_vs_avg_ratio", raw_row)
        assert "above lifetime average" in result
        assert "20%" in result

    def test_month_to_month_contract_reason(self, raw_row):
        assert _format_reason("cat__contract_type_Month-to-month", raw_row) == "month-to-month contract"

    def test_one_year_contract_not_emitted_for_month_to_month_customer(self, raw_row):
        # Customer is Month-to-month, so "One year" reason should NOT fire
        assert _format_reason("cat__contract_type_One year", raw_row) is None

    def test_no_tech_support_reason(self, raw_row):
        assert _format_reason("bool__has_tech_support", raw_row) == "no tech support"

    def test_has_tech_support_reason(self, raw_row):
        raw_row["has_tech_support"] = True
        assert _format_reason("bool__has_tech_support", raw_row) == "has tech support"


class TestTopReasons:
    def test_picks_highest_positive_shap(self, raw_row):
        feature_names = [
            "num__tenure_months",
            "num__monthly_charges",
            "bool__has_tech_support",
        ]
        # SHAP: tech support contributes most, then tenure, monthly_charges is negative
        shap_row = np.array([0.3, -0.1, 0.5])
        reasons = top_reasons(shap_row, feature_names, raw_row, k=3)
        assert reasons[0] == "no tech support"
        assert "short tenure" in reasons[1]
        # monthly_charges is negative — should be excluded
        assert not any("monthly charges of" in r for r in reasons)

    def test_skips_negative_contributors(self, raw_row):
        feature_names = ["num__tenure_months", "num__monthly_charges"]
        shap_row = np.array([-0.5, -0.3])  # all negative
        assert top_reasons(shap_row, feature_names, raw_row) == []

    def test_respects_k_limit(self, raw_row):
        feature_names = [
            "num__tenure_months", "bool__has_tech_support",
            "bool__has_online_security", "cat__contract_type_Month-to-month",
        ]
        shap_row = np.array([0.4, 0.3, 0.2, 0.1])
        assert len(top_reasons(shap_row, feature_names, raw_row, k=2)) == 2

    def test_deduplicates_repeated_reasons(self, raw_row):
        feature_names = ["bool__has_tech_support", "bool__has_tech_support"]
        shap_row = np.array([0.4, 0.3])
        assert top_reasons(shap_row, feature_names, raw_row) == ["no tech support"]
