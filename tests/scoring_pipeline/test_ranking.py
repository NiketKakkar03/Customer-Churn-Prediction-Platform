import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.scoring_pipeline.pipeline import rank_customers


_NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "monthly_vs_avg_ratio", "num_products"]
_BOOLEAN = ["paperless_billing", "has_online_security", "has_tech_support",
            "is_senior_citizen", "has_partner", "has_dependents"]
_CATEGORICAL = ["contract_type", "payment_method", "internet_service"]


@pytest.fixture
def trained_pipeline():
    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), _NUMERIC),
            ("bool", "passthrough", _BOOLEAN),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAL),
        ])),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])
    np.random.seed(0)
    n = 80
    X = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 120, n),
        "total_charges": np.random.uniform(50, 8000, n),
        "monthly_vs_avg_ratio": np.random.uniform(0.5, 1.5, n),
        "num_products": np.random.randint(0, 8, n),
        "paperless_billing": np.random.choice([True, False], n),
        "has_online_security": np.random.choice([True, False], n),
        "has_tech_support": np.random.choice([True, False], n),
        "is_senior_citizen": np.random.choice([True, False], n),
        "has_partner": np.random.choice([True, False], n),
        "has_dependents": np.random.choice([True, False], n),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer"], n),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n),
    })
    # Synthetic churn label: short tenure + month-to-month + no tech support biased to churn
    y = (
        (X["tenure_months"] < 12).astype(int)
        + (X["contract_type"] == "Month-to-month").astype(int)
        + (~X["has_tech_support"]).astype(int)
    ) >= 2
    pipe.fit(X, y.astype(int))
    return pipe


@pytest.fixture
def features():
    return pd.DataFrame({
        "customer_id": ["A", "B", "C", "D", "E"],
        "tenure_months": [2, 60, 1, 30, 5],
        "monthly_charges": [95.0, 50.0, 85.0, 70.0, 100.0],
        "total_charges": [190.0, 3000.0, 85.0, 2100.0, 500.0],
        "monthly_vs_avg_ratio": [1.0, 1.0, 1.0, 1.0, 1.0],
        "num_products": [3, 5, 2, 4, 3],
        "paperless_billing": [True, False, True, False, True],
        "has_online_security": [False, True, False, True, False],
        "has_tech_support": [False, True, False, True, False],
        "is_senior_citizen": [False, False, True, False, False],
        "has_partner": [False, True, False, True, False],
        "has_dependents": [False, True, False, True, False],
        "contract_type": ["Month-to-month", "Two year", "Month-to-month", "One year", "Month-to-month"],
        "payment_method": ["Electronic check", "Bank transfer", "Electronic check", "Mailed check", "Electronic check"],
        "internet_service": ["Fiber optic", "DSL", "Fiber optic", "DSL", "Fiber optic"],
    })


class TestRankCustomers:
    def test_ranks_by_probability_descending(self, trained_pipeline, features):
        ranked = rank_customers(trained_pipeline, features)
        probs = ranked["churn_probability"].values
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    def test_rank_column_is_one_indexed_and_sequential(self, trained_pipeline, features):
        ranked = rank_customers(trained_pipeline, features)
        assert list(ranked["rank"]) == [1, 2, 3, 4, 5]

    def test_top_n_limits_rows(self, trained_pipeline, features):
        ranked = rank_customers(trained_pipeline, features, top_n=3)
        assert len(ranked) == 3
        assert list(ranked["rank"]) == [1, 2, 3]

    def test_top_n_none_returns_all(self, trained_pipeline, features):
        ranked = rank_customers(trained_pipeline, features, top_n=None)
        assert len(ranked) == len(features)

    def test_top_n_larger_than_population_returns_all(self, trained_pipeline, features):
        ranked = rank_customers(trained_pipeline, features, top_n=999)
        assert len(ranked) == len(features)

    def test_shap_reasons_are_non_empty_list(self, trained_pipeline, features):
        ranked = rank_customers(trained_pipeline, features)
        # The top-ranked (highest risk) customer should have at least one reason
        assert isinstance(ranked.iloc[0]["shap_reasons"], list)
        assert len(ranked.iloc[0]["shap_reasons"]) > 0

    def test_shap_reasons_are_strings(self, trained_pipeline, features):
        ranked = rank_customers(trained_pipeline, features)
        for reasons in ranked["shap_reasons"]:
            assert all(isinstance(r, str) for r in reasons)
