"""Train and benchmark four classification pipelines for policyholder churn."""

from pathlib import Path
import sqlite3
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "insurai.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.pkl"

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_premium_rm",
    "annual_value_rm",
    "claims_count",
    "late_payments",
    "support_tickets",
]
CATEGORICAL_FEATURES = ["region", "policy_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_training_data(db_path: Path = DEFAULT_DB_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load cleaned policyholders from SQLite and drop identifiers/leakage columns."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM policyholders;", conn)

    # Drop policy_id (identifier) and churn_reason (post-churn leakage)
    X = df[FEATURE_COLUMNS].copy()
    y = df["churned"].astype(int).copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-free preprocessing transformer for numeric and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_model_specs() -> dict[str, tuple[object, dict[str, list]]]:
    """Return the four benchmarked classifiers and their GridSearchCV parameter grids."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"classifier__C": [0.5, 1.0], "classifier__kernel": ["rbf", "linear"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "classifier__max_depth": [4, 6, 8],
                "classifier__criterion": ["gini", "entropy"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"classifier__n_neighbors": [5, 7, 11]},
        ),
    }


def train_and_benchmark(
    db_path: Path = DEFAULT_DB_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Run stratified split, 5-fold GridSearchCV, evaluate on test set, and save best model."""
    X, y = load_training_data(db_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_models = {}

    for name, (estimator, param_grid) in get_model_specs().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_pipe = grid.best_estimator_
        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Best Params": grid.best_params_,
                "CV ROC-AUC": round(float(grid.best_score_), 4),
                "Test Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "Test F1": round(float(f1_score(y_test, y_pred)), 4),
                "Test ROC-AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
                "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
        )
        fitted_models[name] = best_pipe

    results_df = pd.DataFrame(results).sort_values(
        by=["Test ROC-AUC", "Test F1"], ascending=False
    ).reset_index(drop=True)

    winner_name = results_df.loc[0, "Model"]
    winner_pipeline = fitted_models[winner_name]

    target_path = Path(model_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_pipeline, target_path)

    return {
        "results_df": results_df,
        "winner_name": winner_name,
        "model_path": target_path,
        "fitted_models": fitted_models,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    summary = train_and_benchmark()
    print("Model Benchmark Results:")
    print(
        summary["results_df"][
            ["Model", "CV ROC-AUC", "Test Accuracy", "Test F1", "Test ROC-AUC"]
        ].to_string(index=False)
    )
    print(
        f"\nWinning model: {summary['winner_name']} -> saved to {summary['model_path']}"
    )
