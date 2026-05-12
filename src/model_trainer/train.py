import pathlib
from dataclasses import dataclass

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .metrics import precision_at_k

_CATEGORICAL_COLS = ["contract_type", "payment_method", "internet_service"]
_NUMERIC_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "monthly_vs_avg_ratio", "num_products",
]
_BOOLEAN_COLS = [
    "paperless_billing", "has_online_security", "has_tech_support",
    "is_senior_citizen", "has_partner", "has_dependents",
]
_FEATURE_COLS = _NUMERIC_COLS + _BOOLEAN_COLS + _CATEGORICAL_COLS


@dataclass
class TrainResult:
    model_name: str
    pipeline: Pipeline
    auc_roc: float
    precision_at_50: float
    mlflow_run_id: str


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_COLS),
            ("bool", "passthrough", _BOOLEAN_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAL_COLS),
        ]
    )


def _cv_auc(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> float:
    # TimeSeriesSplit respects temporal order — no shuffling.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        pipeline.fit(X_tr, y_tr)
        prob = pipeline.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            aucs.append(roc_auc_score(y_val, prob))
    return float(np.mean(aucs)) if aucs else 0.0


def train_model(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    mlflow_tracking_uri: str,
    experiment_name: str = "churn-prediction",
) -> list[TrainResult]:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train = train_df[_FEATURE_COLS]
    y_train = train_df["churned"].values
    X_holdout = holdout_df[_FEATURE_COLS]
    y_holdout = holdout_df["churned"].values

    candidates = {
        "xgboost": Pipeline([
            ("pre", _build_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )),
        ]),
        "logistic_regression": Pipeline([
            ("pre", _build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }

    results = []
    for name, pipeline in candidates.items():
        cv_auc = _cv_auc(pipeline, X_train, y_train)

        # Final fit on full training set before holdout evaluation
        pipeline.fit(X_train, y_train)
        holdout_prob = pipeline.predict_proba(X_holdout)[:, 1]
        p50 = precision_at_k(y_holdout, holdout_prob, k=50)
        auc = roc_auc_score(y_holdout, holdout_prob) if len(np.unique(y_holdout)) > 1 else 0.0

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model", name)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_holdout", len(X_holdout))
            if name == "xgboost":
                clf = pipeline.named_steps["clf"]
                mlflow.log_params({
                    "n_estimators": clf.n_estimators,
                    "max_depth": clf.max_depth,
                    "learning_rate": clf.learning_rate,
                })
            mlflow.log_metric("cv_auc_roc", cv_auc)
            mlflow.log_metric("holdout_auc_roc", auc)
            mlflow.log_metric("holdout_precision_at_50", p50)

            results.append(TrainResult(
                model_name=name,
                pipeline=pipeline,
                auc_roc=auc,
                precision_at_50=p50,
                mlflow_run_id=run.info.run_id,
            ))

    return results
