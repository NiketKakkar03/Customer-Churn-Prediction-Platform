import pathlib

import pandas as pd
import streamlit as st

from src.dashboard.data import load_at_risk

st.set_page_config(
    page_title="Churn Risk Dashboard",
    page_icon="📉",
    layout="wide",
)

# ── Drift alert banner ────────────────────────────────────────────────────────
# Placeholder until drift_monitor (issue #8) is complete.
# After that slice lands, this reads from the drift report table instead.
_DRIFT_REPORT = pathlib.Path(__file__).parents[2] / "models" / "drift_report.json"

if _DRIFT_REPORT.exists():
    import json
    report = json.loads(_DRIFT_REPORT.read_text())
    if report.get("drift_detected"):
        st.warning(
            "⚠️ **Feature drift detected** — model predictions may be less reliable. "
            f"Last checked: {report.get('checked_at', 'unknown')}. Retraining may be in progress.",
            icon="⚠️",
        )
else:
    st.info(
        "🔍 **Drift monitoring not yet active.** "
        "Predictions are based on the current champion model.",
        icon="ℹ️",
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Customer Churn Risk — Daily At-Risk List")
st.caption(
    "Ranked by 30-day churn probability. Use this list to prioritise outreach calls today."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    top_n = st.radio("Customers to show", options=[25, 50, 100], index=1)
    st.divider()
    if st.button("🔄 Refresh data"):
        load_at_risk.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
customers, scoring_date = load_at_risk(top_n)

# ── Empty state ───────────────────────────────────────────────────────────────
if not customers:
    st.warning(
        "No predictions available yet. Run the nightly scoring pipeline first:\n\n"
        "```bash\npython -m src.scoring_pipeline.pipeline\n```",
        icon="🚫",
    )
    st.stop()

# ── Scoring date ──────────────────────────────────────────────────────────────
st.caption(f"Scoring date: **{scoring_date}** · Showing top **{len(customers)}** customers")

# ── Summary metrics ───────────────────────────────────────────────────────────
probs = [c["churn_probability"] for c in customers]
col1, col2, col3 = st.columns(3)
col1.metric("Customers flagged", len(customers))
col2.metric("Avg churn probability", f"{sum(probs) / len(probs):.1%}")
col3.metric("High risk (>80%)", sum(1 for p in probs if p > 0.80))

st.divider()

# ── Ranked customer table ─────────────────────────────────────────────────────
df = pd.DataFrame(customers)
df["churn_probability"] = df["churn_probability"].map(lambda p: f"{p:.1%}")
df.columns = ["Rank", "Customer ID", "Churn Probability", "Risk Factors"]

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Churn Probability": st.column_config.TextColumn(width="medium"),
        "Risk Factors": st.column_config.TextColumn(width="large"),
    },
)
