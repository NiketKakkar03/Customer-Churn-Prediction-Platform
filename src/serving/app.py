from typing import Annotated

from fastapi import Depends, FastAPI, Query
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram
from pydantic import BaseModel

from .db import get_conn, fetch_at_risk

app = FastAPI(title="Churn Prediction API")

Instrumentator().instrument(app).expose(app)

# Custom histogram to track the distribution of churn probabilities returned
_PROB_HISTOGRAM = Histogram(
    "churn_probability_returned",
    "Distribution of churn probabilities in API responses",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


class AtRiskCustomer(BaseModel):
    customer_id: str
    churn_probability: float
    rank: int
    reasons: list[str]


class AtRiskResponse(BaseModel):
    customers: list[AtRiskCustomer]
    scoring_date: str | None
    total: int


@app.get("/customers/at-risk", response_model=AtRiskResponse)
def get_at_risk_customers(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    conn=Depends(get_conn),
):
    customers, scoring_date = fetch_at_risk(conn, limit)
    for c in customers:
        _PROB_HISTOGRAM.observe(c["churn_probability"])
    return AtRiskResponse(
        customers=[AtRiskCustomer(**c) for c in customers],
        scoring_date=scoring_date,
        total=len(customers),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
