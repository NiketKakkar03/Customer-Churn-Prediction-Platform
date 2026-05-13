"""
Model quality gate for CI/CD.

Trains a candidate model on the current data, evaluates Precision@50 on
the most-recent 20% time slice, and compares against the current champion.
Exits 0 if the candidate meets or beats the champion; exits 1 otherwise.

Usage:
    python scripts/quality_gate.py
"""
import datetime
import json
import pathlib
import sys

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
CHAMPION_META_PATH = MODELS_DIR / "champion_meta.json"


def _current_champion_p50() -> float:
    if CHAMPION_META_PATH.exists():
        meta = json.loads(CHAMPION_META_PATH.read_text())
        return float(meta.get("precision_at_50", 0.0))
    return 0.0


def main() -> None:
    from src.model_trainer import train

    as_of_date = datetime.date.today()
    champion_p50 = _current_champion_p50()
    print(f"quality_gate: current champion Precision@50 = {champion_p50:.4f}")

    result = train(as_of_date)

    if result is None:
        print(
            f"quality_gate: FAILED — candidate did not meet or beat champion "
            f"(champion={champion_p50:.4f})"
        )
        sys.exit(1)

    print(
        f"quality_gate: PASSED — '{result.model_name}' "
        f"Precision@50={result.precision_at_50:.4f} >= champion={champion_p50:.4f}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
