"""Batch-score policyholders with the trained model and export BI-ready CSV for Power BI / Tableau."""

from pathlib import Path
import sqlite3
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "insurai.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.pkl"
DEFAULT_EXPORT_CSV = PROJECT_ROOT / "data" / "policyholders_scored.csv"

FEATURE_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_premium_rm",
    "annual_value_rm",
    "claims_count",
    "late_payments",
    "support_tickets",
    "region",
    "policy_type",
]


def assign_risk_tier(prob: float) -> str:
    """Map predicted churn probability to a business risk tier."""
    if prob >= 0.50:
        return "High Risk"
    if prob >= 0.30:
        return "Moderate Risk"
    return "Low Risk"


def export_scored_dataset(
    db_path: Path = DEFAULT_DB_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    export_csv: Path = DEFAULT_EXPORT_CSV,
) -> dict[str, object]:
    """Load cleaned data, score with best_churn_model.pkl, and export CSV + SQLite table."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM policyholders;", conn)

    model = joblib.load(model_path)
    probs = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]

    scored = df.copy()
    scored["predicted_churn_prob"] = probs.round(4)
    scored["predicted_churn_pct"] = (probs * 100).round(2)
    scored["predicted_churn_flag"] = (probs >= 0.50).astype(int)
    scored["risk_tier"] = [assign_risk_tier(p) for p in probs]
    scored["expected_premium_at_risk_rm"] = (
        scored["annual_value_rm"] * scored["predicted_churn_prob"]
    ).round(2)

    out_csv = Path(export_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_csv, index=False, encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        scored.to_sql("policyholders_scored", conn, if_exists="replace", index=False)

    return {
        "rows": len(scored),
        "export_csv": out_csv,
        "risk_counts": scored["risk_tier"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    summary = export_scored_dataset()
    print(
        f"Exported {summary['rows']} scored rows -> {summary['export_csv']}\n"
        f"Risk Tier distribution: {summary['risk_counts']}"
    )
