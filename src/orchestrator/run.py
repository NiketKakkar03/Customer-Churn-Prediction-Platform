import datetime
import json
import os
import pathlib

CHAMPION_META_PATH = pathlib.Path(__file__).parents[2] / "models" / "champion_meta.json"


def nightly(as_of_date: datetime.date | None = None) -> None:
    if as_of_date is None:
        as_of_date = datetime.date.today()

    print(f"\n{'=' * 60}")
    print(f"orchestrator: nightly run for {as_of_date}")
    print(f"{'=' * 60}")

    # 1. Compute features
    print("\n[1/4] feature_pipeline")
    from src.feature_pipeline import run as run_features
    run_features(as_of_date)

    # 2. Score all customers
    print("\n[2/4] scoring_pipeline")
    from src.scoring_pipeline import score
    score(as_of_date)

    # 3. Check for feature drift
    print("\n[3/4] drift_monitor")
    from src.drift_monitor import check as check_drift
    drift = check_drift(as_of_date)
    print(
        f"drift_monitor: drift_detected={drift.drift_detected} "
        f"share_drifted={drift.share_drifted_columns:.1%} "
        f"should_retrain={drift.should_retrain}"
    )

    # 4. Conditionally retrain
    retrained = False
    if drift.should_retrain:
        print("\n[4/4] model_trainer (drift + cooldown conditions met)")
        from src.model_trainer import train
        result = train(as_of_date)
        if result:
            retrained = True
            # Re-score with the new champion
            print("\n[4b] scoring_pipeline (re-scoring with new champion)")
            score(as_of_date)
    else:
        reason = (
            "no drift detected" if not drift.drift_detected
            else "cooldown period not yet elapsed"
        )
        print(f"\n[4/4] model_trainer skipped — {reason}")

    champion_p50 = _read_champion_p50()
    _push_metrics(drift.drift_detected, drift.share_drifted_columns, retrained, champion_p50)

    print(f"\n{'=' * 60}")
    print(f"orchestrator: nightly run complete for {as_of_date}")
    print(f"{'=' * 60}\n")


def _read_champion_p50() -> float:
    if CHAMPION_META_PATH.exists():
        meta = json.loads(CHAMPION_META_PATH.read_text())
        return float(meta.get("precision_at_50", 0.0))
    return 0.0


def _push_metrics(
    drift_detected: bool,
    share_drifted: float,
    retrained: bool,
    champion_p50: float,
) -> None:
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        Gauge("churn_drift_detected", "1 if feature drift detected", registry=registry).set(
            1 if drift_detected else 0
        )
        Gauge("churn_drift_share", "Share of drifted feature columns", registry=registry).set(
            share_drifted
        )
        Gauge("churn_retraining_triggered", "1 if retraining ran this cycle", registry=registry).set(
            1 if retrained else 0
        )
        Gauge("churn_champion_precision_at_50", "Champion model Precision@50", registry=registry).set(
            champion_p50
        )

        url = os.getenv("PUSHGATEWAY_URL", "localhost:9091")
        push_to_gateway(url, job="churn_orchestrator", registry=registry)
        print(f"orchestrator: metrics pushed to pushgateway at {url}")
    except Exception as exc:
        print(f"orchestrator: metric push skipped — {exc}")


if __name__ == "__main__":
    nightly()
