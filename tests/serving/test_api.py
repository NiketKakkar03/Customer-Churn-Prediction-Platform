import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.db import get_conn

_FAKE_ROWS = [
    {"customer_id": "C001", "churn_probability": 0.91, "rank": 1, "reasons": ["short tenure (2 months)", "month-to-month contract"]},
    {"customer_id": "C002", "churn_probability": 0.84, "rank": 2, "reasons": ["fiber optic internet"]},
    {"customer_id": "C003", "churn_probability": 0.77, "rank": 3, "reasons": ["no tech support", "no online security"]},
]


def _make_fake_conn(rows):
    class FakeCursor:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def execute(self_, _sql, args=()): self_._limit = args[0] if args else len(rows)
        def fetchall(self_):
            if not rows:
                return []
            limit = getattr(self_, "_limit", len(rows))
            return [
                (r["customer_id"], r["churn_probability"], r["rank"], r["reasons"], "2026-05-12")
                for r in rows[:limit]
            ]

    class FakeConn:
        def cursor(self): return FakeCursor()
        def close(self): pass

    def _get():
        yield FakeConn()

    return _get


@pytest.fixture
def client():
    app.dependency_overrides[get_conn] = _make_fake_conn(_FAKE_ROWS)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def empty_client():
    app.dependency_overrides[get_conn] = _make_fake_conn([])
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestAtRiskEndpoint:
    def test_happy_path_returns_200(self, client):
        resp = client.get("/customers/at-risk")
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/customers/at-risk").json()
        assert "customers" in data
        assert "scoring_date" in data
        assert "total" in data

    def test_customers_have_required_fields(self, client):
        customers = client.get("/customers/at-risk").json()["customers"]
        for c in customers:
            assert "customer_id" in c
            assert "churn_probability" in c
            assert "rank" in c
            assert "reasons" in c

    def test_customers_ordered_by_rank(self, client):
        customers = client.get("/customers/at-risk").json()["customers"]
        ranks = [c["rank"] for c in customers]
        assert ranks == sorted(ranks)

    def test_limit_parameter_respected(self, client):
        resp = client.get("/customers/at-risk?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["customers"]) == 2

    def test_limit_defaults_to_50_when_omitted(self, client):
        resp = client.get("/customers/at-risk")
        assert resp.status_code == 200
        # Our fake only has 3 rows, so all 3 are returned (< default 50)
        assert len(resp.json()["customers"]) == 3

    def test_total_matches_customers_length(self, client):
        data = client.get("/customers/at-risk").json()
        assert data["total"] == len(data["customers"])

    def test_scoring_date_in_response(self, client):
        data = client.get("/customers/at-risk").json()
        assert data["scoring_date"] == "2026-05-12"

    def test_reasons_are_list_of_strings(self, client):
        customers = client.get("/customers/at-risk").json()["customers"]
        for c in customers:
            assert isinstance(c["reasons"], list)
            assert all(isinstance(r, str) for r in c["reasons"])


class TestEmptyTable:
    def test_returns_200_not_error(self, empty_client):
        assert empty_client.get("/customers/at-risk").status_code == 200

    def test_returns_empty_list(self, empty_client):
        data = empty_client.get("/customers/at-risk").json()
        assert data["customers"] == []
        assert data["total"] == 0
        assert data["scoring_date"] is None


class TestValidation:
    def test_limit_zero_is_rejected(self, client):
        assert client.get("/customers/at-risk?limit=0").status_code == 422

    def test_limit_above_max_is_rejected(self, client):
        assert client.get("/customers/at-risk?limit=501").status_code == 422


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
