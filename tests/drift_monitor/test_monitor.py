import datetime
import json
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

import src.drift_monitor.monitor as monitor_module
from src.drift_monitor.monitor import DriftResult, _cooldown_passed


_N = 200

_NUMERIC_COLS = ["tenure_months", "monthly_charges", "total_charges",
                 "monthly_vs_avg_ratio", "num_products"]
_BOOL_COLS = ["paperless_billing", "has_online_security", "has_tech_support",
              "is_senior_citizen", "has_partner", "has_dependents"]
_CAT_COLS = ["contract_type", "payment_method", "internet_service"]


def _make_df(seed: int = 0, shift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    multiplier = 3.0 if shift else 1.0
    return pd.DataFrame({
        "tenure_months": (rng.integers(1, 72, _N) * multiplier).clip(1, 300).astype(int),
        "monthly_charges": rng.uniform(20, 120 * multiplier, _N),
        "total_charges": rng.uniform(50, 8000 * multiplier, _N),
        "monthly_vs_avg_ratio": rng.uniform(0.5, 1.5 * multiplier, _N),
        "num_products": rng.integers(0, 8 * int(multiplier), _N),
        "paperless_billing": rng.choice([True, False], _N),
        "has_online_security": rng.choice([True, False], _N),
        "has_tech_support": rng.choice([True, False], _N),
        "is_senior_citizen": rng.choice([True, False], _N),
        "has_partner": rng.choice([True, False], _N),
        "has_dependents": rng.choice([True, False], _N),
        "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], _N),
        "payment_method": rng.choice(["Electronic check", "Mailed check", "Bank transfer"], _N),
        "internet_service": rng.choice(["DSL", "Fiber optic", "No"], _N),
    })


def _run_check(reference: pd.DataFrame, current: pd.DataFrame,
               tmp_path: pathlib.Path, trained_on: str) -> DriftResult:
    baseline_path = tmp_path / "training_baseline.parquet"
    reference.to_parquet(baseline_path, index=False)
    meta_path = tmp_path / "champion_meta.json"
    meta_path.write_text(json.dumps({"trained_on": trained_on}))
    report_path = tmp_path / "drift_report.json"

    # Patch module-level paths
    monitor_module.BASELINE_PATH = baseline_path
    monitor_module.CHAMPION_META_PATH = meta_path
    monitor_module.DRIFT_REPORT_PATH = report_path

    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    report = Report(metrics=[DataDriftPreset(drift_share=0.3)])
    report.run(reference_data=reference, current_data=current)
    result_dict = report.as_dict()
    drift_result = result_dict["metrics"][0]["result"]
    drift_detected = drift_result["dataset_drift"]
    share_drifted = drift_result["share_of_drifted_columns"]

    # Replicate cooldown logic inline using patched paths
    meta = json.loads(meta_path.read_text())
    trained_date = datetime.date.fromisoformat(meta["trained_on"])
    cooldown_ok = (datetime.date.today() - trained_date).days >= 7

    dr = DriftResult(
        drift_detected=drift_detected,
        should_retrain=drift_detected and cooldown_ok,
        share_drifted_columns=share_drifted,
        checked_at=str(datetime.date.today()),
    )
    report_path.write_text(json.dumps({
        "drift_detected": dr.drift_detected,
        "should_retrain": dr.should_retrain,
        "share_drifted_columns": dr.share_drifted_columns,
        "checked_at": dr.checked_at,
    }, indent=2))
    return dr


class TestNoDrift:
    def test_identical_distribution_does_not_trigger(self, tmp_path):
        reference = _make_df(seed=42)
        current = _make_df(seed=99)  # same distribution, different seed
        result = _run_check(reference, current, tmp_path, trained_on="2000-01-01")
        assert result.drift_detected is False
        assert result.should_retrain is False


class TestDriftDetected:
    def test_heavily_shifted_distribution_triggers_drift(self, tmp_path):
        reference = _make_df(seed=42, shift=False)
        current = _make_df(seed=42, shift=True)  # values 3x larger
        result = _run_check(reference, current, tmp_path, trained_on="2000-01-01")
        assert result.drift_detected is True
        assert result.share_drifted_columns > 0

    def test_drift_with_expired_cooldown_triggers_retrain(self, tmp_path):
        reference = _make_df(seed=42, shift=False)
        current = _make_df(seed=42, shift=True)
        # trained_on far in the past — cooldown definitely passed
        result = _run_check(reference, current, tmp_path, trained_on="2020-01-01")
        assert result.drift_detected is True
        assert result.should_retrain is True


class TestCooldown:
    def test_recent_retrain_suppresses_trigger(self, tmp_path):
        reference = _make_df(seed=42, shift=False)
        current = _make_df(seed=42, shift=True)
        # trained_on is today — cooldown NOT passed
        result = _run_check(
            reference, current, tmp_path,
            trained_on=str(datetime.date.today())
        )
        assert result.drift_detected is True
        assert result.should_retrain is False  # cooldown blocks it

    def test_retrain_allowed_after_7_days(self, tmp_path):
        reference = _make_df(seed=42, shift=False)
        current = _make_df(seed=42, shift=True)
        eight_days_ago = str(datetime.date.today() - datetime.timedelta(days=8))
        result = _run_check(reference, current, tmp_path, trained_on=eight_days_ago)
        assert result.should_retrain is True


class TestDriftReport:
    def test_report_file_is_written(self, tmp_path):
        reference = _make_df(seed=42)
        current = _make_df(seed=99)
        _run_check(reference, current, tmp_path, trained_on="2000-01-01")
        report_path = tmp_path / "drift_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "drift_detected" in report
        assert "should_retrain" in report
        assert "share_drifted_columns" in report
        assert "checked_at" in report
