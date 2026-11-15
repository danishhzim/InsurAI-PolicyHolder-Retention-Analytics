"""ETL pipeline to clean raw policyholder records and load them into SQLite."""

from pathlib import Path
import sqlite3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_CSV = PROJECT_ROOT / "Dataset" / "raw_policyholders.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "insurai.db"
TABLE_NAME = "policyholders"

INTEGER_COLUMNS = [
    "age",
    "tenure_months",
    "claims_count",
    "late_payments",
    "support_tickets",
    "churned",
]

TABLE_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    policy_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL,
    region TEXT NOT NULL,
    policy_type TEXT NOT NULL,
    tenure_months INTEGER NOT NULL,
    monthly_premium_rm REAL NOT NULL,
    annual_value_rm REAL NOT NULL,
    claims_count INTEGER NOT NULL,
    late_payments INTEGER NOT NULL,
    support_tickets INTEGER NOT NULL,
    churned INTEGER NOT NULL,
    churn_reason TEXT NOT NULL
);
"""


def extract_raw_data(csv_path: Path = DEFAULT_RAW_CSV) -> pd.DataFrame:
    """Read raw policyholder records from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    return pd.read_csv(path)


def transform_policyholders(df: pd.DataFrame) -> pd.DataFrame:
    """Clean anomalies, impute missing values, and engineer annual_value_rm."""
    cleaned = df.copy()

    # Normalize string columns to eliminate whitespace and casing inconsistencies
    cleaned["policy_id"] = cleaned["policy_id"].astype(str).str.strip()
    cleaned["region"] = cleaned["region"].astype(str).str.strip().str.title()
    cleaned["policy_type"] = cleaned["policy_type"].astype(str).str.strip()

    # Impute missing late_payments with 0 before casting integer columns
    cleaned["late_payments"] = pd.to_numeric(
        cleaned["late_payments"], errors="coerce"
    ).fillna(0)
    for col in INTEGER_COLUMNS:
        cleaned[col] = cleaned[col].astype(int)

    cleaned["monthly_premium_rm"] = cleaned["monthly_premium_rm"].astype(float).round(2)
    cleaned["annual_value_rm"] = (cleaned["monthly_premium_rm"] * 12).round(2)

    # Retained customers (churned == 0) have blank churn_reason; fill for zero-NULL SQL storage
    cleaned["churn_reason"] = (
        cleaned["churn_reason"]
        .fillna("Not Applicable")
        .astype(str)
        .str.strip()
        .replace("", "Not Applicable")
    )

    ordered_columns = [
        "policy_id",
        "age",
        "region",
        "policy_type",
        "tenure_months",
        "monthly_premium_rm",
        "annual_value_rm",
        "claims_count",
        "late_payments",
        "support_tickets",
        "churned",
        "churn_reason",
    ]
    return cleaned[ordered_columns]


def load_to_sqlite(
    df: pd.DataFrame,
    db_path: Path = DEFAULT_DB_PATH,
    table_name: str = TABLE_NAME,
) -> dict[str, int]:
    """Create the SQLite table with a primary key, load data, and return row/null counts."""
    target_db = Path(db_path)
    target_db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target_db) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.execute(TABLE_SCHEMA_SQL)
        df.to_sql(table_name, conn, if_exists="append", index=False)

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
        null_checks = " + ".join(
            f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)" for col in df.columns
        )
        null_count = conn.execute(
            f"SELECT {null_checks} FROM {table_name};"
        ).fetchone()[0]

    return {"row_count": int(row_count), "null_count": int(null_count)}


def run_etl(
    csv_path: Path = DEFAULT_RAW_CSV,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Execute full Extract -> Transform -> Load pipeline."""
    raw_df = extract_raw_data(csv_path)
    cleaned_df = transform_policyholders(raw_df)
    stats = load_to_sqlite(cleaned_df, db_path)
    return stats


if __name__ == "__main__":
    result = run_etl()
    print(
        f"ETL complete -> {DEFAULT_DB_PATH} | "
        f"Rows loaded: {result['row_count']} | "
        f"Total NULLs: {result['null_count']}"
    )
