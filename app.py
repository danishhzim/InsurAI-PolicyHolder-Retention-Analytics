"""Interactive Streamlit BI dashboard and live churn risk simulator for InsurAI."""

from pathlib import Path
import sqlite3
import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "insurai.db"
MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.pkl"

st.set_page_config(
    page_title="InsurAI | Policyholder Retention & Risk Analytics",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_policyholders() -> pd.DataFrame:
    """Load cleaned policyholder records from SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM policyholders;", conn)


@st.cache_resource
def load_model():
    """Load the serialized Scikit-Learn churn prediction pipeline."""
    return joblib.load(MODEL_PATH)


def query_filtered_sql(regions: list[str], policy_types: list[str], limit: int = 200) -> pd.DataFrame:
    """Query filtered records directly from insurai.db using parameterized SQL."""
    region_placeholders = ",".join("?" for _ in regions)
    policy_placeholders = ",".join("?" for _ in policy_types)
    sql = f"""
        SELECT
            policy_id, age, region, policy_type, tenure_months,
            monthly_premium_rm, annual_value_rm, claims_count,
            late_payments, support_tickets, churned, churn_reason
        FROM policyholders
        WHERE region IN ({region_placeholders})
          AND policy_type IN ({policy_placeholders})
        ORDER BY annual_value_rm DESC
        LIMIT ?;
    """
    params = [*regions, *policy_types, limit]
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_retention_recommendation(prob: float, late_payments: int, support_tickets: int) -> tuple[str, str]:
    """Return risk tier label and tailored retention action."""
    if prob >= 0.50:
        tier = "High Risk"
        if late_payments >= 2:
            action = "Offer flexible billing grace period + proactive retention call before renewal."
        elif support_tickets >= 2:
            action = "Escalate to senior account manager to resolve open service complaints."
        else:
            action = "Offer loyalty renewal discount (5-10%) and review coverage fit."
    elif prob >= 0.30:
        tier = "Moderate Risk"
        action = "Schedule automated renewal check-in and highlight policy benefits."
    else:
        tier = "Low Risk"
        action = "Standard renewal workflow; eligible for cross-sell or referral outreach."
    return tier, action


def main() -> None:
    st.title("InsurAI: Policyholder Retention & Risk Analytics")
    st.caption(
        "End-to-end churn analytics and premium-at-risk simulator powered by SQLite & Scikit-Learn."
    )

    if not DB_PATH.exists() or not MODEL_PATH.exists():
        st.error(
            "Missing `data/insurai.db` or `models/best_churn_model.pkl`. "
            "Run `python src/etl_pipeline.py` and `python src/train_model.py` first."
        )
        st.stop()

    df = load_policyholders()
    model = load_model()

    # Sidebar filters
    st.sidebar.header("Portfolio Filters")
    all_regions = sorted(df["region"].unique().tolist())
    all_policies = sorted(df["policy_type"].unique().tolist())

    selected_regions = st.sidebar.multiselect(
        "Region", options=all_regions, default=all_regions
    )
    selected_policies = st.sidebar.multiselect(
        "Policy Type", options=all_policies, default=all_policies
    )

    if not selected_regions or not selected_policies:
        st.warning("Select at least one Region and one Policy Type in the sidebar to view metrics.")
        st.stop()

    filtered = df[
        df["region"].isin(selected_regions) & df["policy_type"].isin(selected_policies)
    ]

    # Executive KPI Header
    total_policies = len(filtered)
    churn_rate_pct = filtered["churned"].mean() * 100
    revenue_at_risk = filtered.loc[filtered["churned"] == 1, "annual_value_rm"].sum()
    avg_tenure = filtered["tenure_months"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Policies", f"{total_policies:,}")
    col2.metric("Churn Rate", f"{churn_rate_pct:.2f}%")
    col3.metric("Annual Premium at Risk", f"RM {revenue_at_risk:,.2f}")
    col4.metric("Avg. Tenure", f"{avg_tenure:.1f} mo")

    st.divider()

    tab_insights, tab_simulator, tab_explorer = st.tabs(
        ["Business Insights", "Live ML Risk Simulator", "SQL Data Explorer"]
    )

    with tab_insights:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Churn Rate (%) by Policy Type")
            policy_chart = (
                filtered.groupby("policy_type")["churned"]
                .mean()
                .mul(100)
                .round(2)
                .sort_values(ascending=False)
            )
            st.bar_chart(policy_chart)

        with c2:
            st.subheader("Churn Rate (%) by Late Payments")
            late_chart = (
                filtered.groupby("late_payments")["churned"]
                .mean()
                .mul(100)
                .round(2)
            )
            st.bar_chart(late_chart)

        st.subheader("Top Stated Churn Reasons")
        churned_only = filtered[filtered["churned"] == 1]
        if churned_only.empty:
            st.info("No churned policyholders in the current filter selection.")
        else:
            reason_summary = (
                churned_only.groupby("churn_reason", as_index=False)
                .agg(
                    churned_count=("policy_id", "count"),
                    lost_annual_value_rm=("annual_value_rm", "sum"),
                )
                .sort_values("churned_count", ascending=False)
            )
            reason_summary["lost_annual_value_rm"] = reason_summary["lost_annual_value_rm"].round(2)
            st.dataframe(reason_summary, use_container_width=True, hide_index=True)

    with tab_simulator:
        st.subheader("Simulate Policyholder Churn Risk")
        with st.form("churn_simulator_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                age = st.number_input("Age", min_value=18, max_value=85, value=34)
                region = st.selectbox("Region", options=all_regions)
                policy_type = st.selectbox("Policy Type", options=all_policies)
            with f2:
                tenure_months = st.number_input("Tenure (Months)", min_value=1, max_value=240, value=8)
                monthly_premium_rm = st.number_input(
                    "Monthly Premium (RM)", min_value=30.0, max_value=2000.0, value=380.0, step=10.0
                )
                claims_count = st.number_input("Claims Count", min_value=0, max_value=20, value=1)
            with f3:
                late_payments = st.number_input("Late Payments (12 mo)", min_value=0, max_value=12, value=3)
                support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=4)

            submitted = st.form_submit_button("Predict Churn Risk", type="primary")

        if submitted:
            annual_value_rm = round(monthly_premium_rm * 12, 2)
            input_df = pd.DataFrame(
                [
                    {
                        "age": age,
                        "tenure_months": tenure_months,
                        "monthly_premium_rm": monthly_premium_rm,
                        "annual_value_rm": annual_value_rm,
                        "claims_count": claims_count,
                        "late_payments": late_payments,
                        "support_tickets": support_tickets,
                        "region": region,
                        "policy_type": policy_type,
                    }
                ]
            )
            churn_prob = float(model.predict_proba(input_df)[0, 1])
            risk_tier, recommendation = get_retention_recommendation(
                churn_prob, late_payments, support_tickets
            )

            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Churn Probability", f"{churn_prob * 100:.1f}%")
            r2.metric("Risk Tier", risk_tier)
            r3.metric("Annual Premium Exposure", f"RM {annual_value_rm:,.2f}")
            st.info(f"**Recommended Action:** {recommendation}")

    with tab_explorer:
        st.subheader("Live SQL Explorer (`data/insurai.db`)")
        row_limit = st.slider("Max rows to display", min_value=20, max_value=500, value=100, step=20)
        sql_df = query_filtered_sql(selected_regions, selected_policies, limit=row_limit)
        st.dataframe(sql_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
