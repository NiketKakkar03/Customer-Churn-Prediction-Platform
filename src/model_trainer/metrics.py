import numpy as np
import pandas as pd


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 50) -> float:
    """
    Fraction of true churners in the top-k customers ranked by churn probability.
    This maps directly to the business output: of the top-50 customers we flag,
    how many actually churned?
    """
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def time_aware_split(df: pd.DataFrame, holdout_frac: float = 0.20):
    """
    Split df into train and holdout using tenure_months as a time proxy.
    Customers with lower tenure are newer — the holdout is the most recent
    (lowest-tenure) fraction. Train on older, test on newer.
    """
    df_sorted = df.sort_values("tenure_months", ascending=False).reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - holdout_frac))
    train = df_sorted.iloc[:cutoff].copy()
    holdout = df_sorted.iloc[cutoff:].copy()
    return train, holdout
