import datetime
import json
import os
import pathlib

import joblib
from dotenv import load_dotenv

from .data import load_features
from .metrics import time_aware_split
from .train import train_model, TrainResult

load_dotenv()

MODELS_DIR = pathlib.Path(__file__).parents[2] / "models"
CHAMPION_PATH = MODELS_DIR / "champion.joblib"
CHAMPION_META_PATH = MODELS_DIR / "champion_meta.json"


def train(as_of_date: datetime.date | None = None) -> TrainResult | None:
    if as_of_date is None:
        as_of_date = datetime.date.today()

    MODELS_DIR.mkdir(exist_ok=True)

    df = load_features(as_of_date)
    train_df, holdout_df = time_aware_split(df, holdout_frac=0.20)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    results = train_model(train_df, holdout_df, mlflow_tracking_uri=mlflow_uri)

    best = max(results, key=lambda r: r.precision_at_50)
    champion_p50 = _load_champion_p50()

    if best.precision_at_50 >= champion_p50:
        joblib.dump(best.pipeline, CHAMPION_PATH)
        _save_champion_meta(best, as_of_date)
        _save_training_baseline(train_df)
        print(
            f"model_trainer: promoted '{best.model_name}' "
            f"Precision@50={best.precision_at_50:.4f} "
            f"(prev champion={champion_p50:.4f})"
        )
        return best
    else:
        print(
            f"model_trainer: no promotion — best candidate '{best.model_name}' "
            f"Precision@50={best.precision_at_50:.4f} < champion={champion_p50:.4f}"
        )
        return None


def _load_champion_p50() -> float:
    if CHAMPION_META_PATH.exists():
        meta = json.loads(CHAMPION_META_PATH.read_text())
        return float(meta.get("precision_at_50", 0.0))
    return 0.0


def _save_training_baseline(train_df) -> None:
    import pandas as pd
    from .train import _FEATURE_COLS
    baseline_path = MODELS_DIR / "training_baseline.parquet"
    train_df[_FEATURE_COLS].to_parquet(baseline_path, index=False)


def _save_champion_meta(result: TrainResult, as_of_date: datetime.date) -> None:
    meta = {
        "model_name": result.model_name,
        "precision_at_50": result.precision_at_50,
        "auc_roc": result.auc_roc,
        "mlflow_run_id": result.mlflow_run_id,
        "trained_on": str(as_of_date),
    }
    CHAMPION_META_PATH.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    train()
