CREATE TABLE IF NOT EXISTS churn_predictions (
    customer_id        TEXT        NOT NULL,
    scoring_date       DATE        NOT NULL,
    churn_probability  NUMERIC(6, 5) NOT NULL,
    rank               INTEGER     NOT NULL,
    shap_reasons       JSONB       NOT NULL DEFAULT '[]',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (customer_id, scoring_date)
);

CREATE INDEX IF NOT EXISTS idx_churn_predictions_date_rank
    ON churn_predictions (scoring_date, rank);
