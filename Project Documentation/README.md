# InsurAI: Policyholder Retention & Risk Analytics Pipeline

End-to-end applied data science pipeline for predicting insurance policyholder churn and quantifying annual recurring premium at risk (RM), aligned with the **IBM Data Science Methodology (CRISP-DM)**.

> **Note on Data:** The `Dataset/raw_policyholders.csv` dataset (1,500 records) is synthetically generated from domain assumptions to simulate realistic data quality issues (inconsistent region casing/whitespace and missing values) for end-to-end pipeline development.

## Project Structure

```text
├── assets/                                      # Saved charts & dashboard screenshots
├── data/                                        # Normalized SQLite database (insurai.db)
├── Dataset/                                     # Raw synthetic CSV & data generation script
├── models/                                      # Serialized Scikit-Learn pipeline (.pkl)
├── notebooks/                                   # Modular CRISP-DM analytical notebooks
├── src/                                         # Production ETL & model training scripts
├── app.py                                       # Interactive Streamlit BI dashboard
├── requirements.txt                             # Python dependencies
└── README.md
```

## Environment Setup

```powershell
# 1. Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt
```

## Pipeline Stages

1. **ETL & SQLite Storage:** Clean `Dataset/raw_policyholders.csv` and load into `data/insurai.db`.
2. **SQL & Visual EDA:** Analyze regional and product-tier churn drivers and revenue at risk.
3. **Geospatial Analytics:** Map Malaysian regional churn hotspots using `Folium`.
4. **ML Benchmarking:** Compare Logistic Regression, SVM, Decision Tree, and KNN with `GridSearchCV` inside leakage-free `Pipeline`s.
5. **Streamlit Deployment:** Interactive executive KPI dashboard and live churn risk simulator (`app.py`).
