# InsurAI: Policyholder Retention & Risk Analytics Pipeline
**End-to-End Applied Data Science Capstone Documentation**  
*Aligned with the IBM Data Science Professional Certificate Methodology*

---

## 1. Executive Summary & Project Overview
**InsurAI** is an end-to-end data engineering, predictive machine learning, and business intelligence solution designed for the insurance and FinTech domain. Built strictly following the **IBM Data Science Methodology (CRISP-DM)**, the project ingests raw policyholder data, cleans and normalizes anomalies into a relational SQL database, uncovers behavioral churn drivers through SQL and geospatial exploratory data analysis (EDA), benchmarks four supervised classification algorithms, and deploys the optimal model into an interactive web dashboard.

* **Project Author:** Danish Hazim Bin Mohamed Zahirulfaiza
* **Domain:** InsurTech / Financial Services / Predictive Analytics
* **Methodology:** IBM Data Science Methodology (CRISP-DM)
* **Primary Objectives:**
  1. Automate data ingestion, cleaning, and relational storage via a reproducible Python + SQL ETL pipeline.
  2. Identify key demographic, behavioral, and policy-level drivers of customer churn using SQL aggregations and statistical/geospatial visualizations.
  3. Benchmark multiple classification models (`Logistic Regression`, `SVM`, `Decision Tree`, `KNN`) using cross-validated hyperparameter tuning (`GridSearchCV`).
  4. Translate technical model outputs into actionable retention strategies via a live interactive dashboard.

---

## 2. System Architecture & Repository Structure

### 2.1 High-Level Data Flow
```text
[Raw Policyholder Data] 
        │
        ▼ (Pandas / NumPy ETL & Cleaning)
[Relational Database: insurai.db (SQLite)]
        │
        ├──► [SQL Aggregations & Statistical EDA (Seaborn / Matplotlib)]
        ├──► [Geospatial Risk Mapping (Folium - Malaysia Regions)]
        └──► [ML Pipeline: StandardScaler + OneHotEncoder + GridSearchCV]
                    │
                    ▼ (Serialized Artifact: best_churn_model.pkl)
        [Interactive Web Application: Streamlit Dashboard (app.py)]
```

### 2.2 Standardized Repository Structure
Organize the GitHub repository (`InsurAI-IBM-Capstone-Analytics`) cleanly so technical recruiters and hiring managers can inspect both modular analytical notebooks and production scripts:

```text
InsurAI-IBM-Capstone-Analytics/
├── data/
│   └── insurai.db                               # Normalized relational SQLite database
├── models/
│   └── best_churn_model.pkl                     # Serialized top-performing Scikit-Learn pipeline
├── notebooks/
│   ├── 01_Data_Collection_and_Wrangling.ipynb   # ETL, missing value imputation & feature engineering
│   ├── 02_SQL_Exploratory_Data_Analysis.ipynb   # Relational SQL queries & statistical hypothesis checks
│   ├── 03_Geospatial_and_Visual_Analytics.ipynb # Seaborn distributions & Folium Malaysia risk map
│   └── 04_ML_Classification_Benchmark.ipynb     # Multi-model training, GridSearchCV & ROC-AUC evaluation
├── src/
│   ├── etl_pipeline.py                          # Automated production script for database generation
│   └── train_model.py                           # Automated production script for model retraining
├── app.py                                       # Interactive Streamlit BI dashboard & live risk simulator
├── requirements.txt                             # Reproducible Python environment dependencies
└── README.md                                    # Executive summary, setup guide & key findings
```

---

## 3. Technology Stack & Tools Used

| Layer / Phase | Tools & Libraries | Purpose & IBM Module Alignment |
| :--- | :--- | :--- |
| **Programming & Environment** | Python 3.10+, Jupyter Notebooks, VS Code / PyCharm, Git & GitHub | Core development, modular experimentation, and version control (*IBM Tools for Data Science*). |
| **Data Wrangling & ETL** | `Pandas`, `NumPy` | Data ingestion, string normalization, null imputation, and feature engineering (*IBM Data Analysis with Python*). |
| **Relational Database & Querying** | `SQLite3`, SQL (`GROUP BY`, `HAVING`, `CASE WHEN`) | Schema design, relational storage, and analytical SQL aggregations (*IBM Databases & SQL for Data Science*). |
| **Data & Geospatial Visualization** | `Matplotlib`, `Seaborn`, `Folium` | Statistical distribution plots, correlation heatmaps, and interactive Malaysian regional maps (*IBM Data Visualization with Python*). |
| **Machine Learning & Evaluation** | `Scikit-Learn` (`Pipeline`, `ColumnTransformer`, `GridSearchCV`), `Joblib` | Preprocessing pipelines, multi-model classification benchmarking, and model serialization (*IBM Machine Learning with Python*). |
| **Deployment & BI Interface** | `Streamlit`, Streamlit Community Cloud | Interactive executive KPI dashboard, SQL data explorer, and real-time churn probability simulator (*IBM Applied Data Science Capstone*). |

---

## 4. End-to-End Execution Process (IBM Methodology)

### Phase 1: Business Understanding & Data Schema Design
* **Problem Statement:** Acquiring a new insurance policyholder costs significantly more than retaining an existing one. Management needs an automated system to flag high-risk policies before renewal and quantify annual recurring premium at risk.
* **Database Schema (`policyholders` table):**
  * `policy_id` (TEXT, Primary Key): Unique identifier (`POL-10000` series).
  * `age` (INTEGER): Policyholder age (21–70).
  * `region` (TEXT): Malaysian state/territory (`Kuala Lumpur`, `Selangor`, `Johor`, `Penang`, `Sarawak`).
  * `policy_type` (TEXT): Product tier (`Medical & Health`, `Life Protection`, `Motor Comprehensive`, `Investment-Linked`).
  * `tenure_months` (INTEGER): Duration of active coverage in months.
  * `monthly_premium_rm` (REAL): Monthly recurring billing amount in Ringgit Malaysia (RM).
  * `annual_value_rm` (REAL): Engineered feature (`monthly_premium_rm * 12`).
  * `claims_count` (INTEGER): Total claims submitted during tenure.
  * `late_payments` (INTEGER): Number of overdue billing cycles in the past 12 months.
  * `support_tickets` (INTEGER): Number of customer service complaints/inquiries logged.
  * `churned` (INTEGER): Target binary variable (`1` = Churned, `0` = Retained).

### Phase 2: Data Collection & Wrangling (`01_Data_Collection_and_Wrangling.ipynb`)
1. **Extract:** Ingest raw policyholder records containing real-world anomalies (e.g., trailing whitespace and inconsistent capitalization in region names like `" kuala lumpur "`, and `NaN` values in `late_payments`).
2. **Transform:**
   * Standardize categorical strings via `.str.strip().str.title()`.
   * Impute missing `late_payments` values with `0` (median/mode imputation) and cast to integer.
   * Derive `annual_value_rm` to support financial risk exposure calculations.
3. **Load:** Commit the cleaned dataset into `data/insurai.db` using `sqlite3` with explicit primary key constraints.

### Phase 3: SQL & Visual Exploratory Data Analysis (`02_SQL_Exploratory_Data_Analysis.ipynb` & `03_Geospatial_and_Visual_Analytics.ipynb`)
1. **SQL Analytical Queries:** Execute structured SQL queries to segment churn rates and financial exposure across regions and product tiers:
   ```sql
   SELECT 
       region,
       policy_type,
       COUNT(*) AS total_customers,
       SUM(churned) AS churned_customers,
       ROUND(AVG(churned) * 100, 2) AS churn_rate_pct,
       ROUND(SUM(CASE WHEN churned = 1 THEN annual_value_rm ELSE 0 END), 2) AS revenue_at_risk_rm
   FROM policyholders
   GROUP BY region, policy_type
   ORDER BY churn_rate_pct DESC;
   ```
2. **Statistical & Geospatial EDA:**
   * Generate bar charts and line plots (`Seaborn`) illustrating how churn spikes when `tenure_months < 12` or `late_payments >= 2`.
   * Plot an interactive `Folium` choropleth/marker map across Kuala Lumpur, Selangor, Johor, Penang, and Sarawak displaying regional churn hotspots.

### Phase 4: Predictive Modeling & Benchmarking (`04_ML_Classification_Benchmark.ipynb`)
1. **Leakage-Free Preprocessing Pipeline:**
   * Apply `StandardScaler` to numerical features and `OneHotEncoder(handle_unknown="ignore")` to categorical features inside a `ColumnTransformer`.
   * Split data using stratified sampling (`80%` train / `20%` test, `random_state=42`) to preserve target class balance.
2. **Multi-Algorithm Comparison (`GridSearchCV`, 5-Fold CV):**
   * **Logistic Regression:** Tuned regularization strength `C` (`[0.1, 1.0, 10.0]`).
   * **Support Vector Machine (SVM):** Tuned `C` (`[0.5, 1.0]`) and kernel (`['rbf', 'linear']`).
   * **Decision Tree Classifier:** Tuned `max_depth` (`[4, 6, 8]`) and split criterion (`['gini', 'entropy']`).
   * **K-Nearest Neighbors (KNN):** Tuned `n_neighbors` (`[5, 7, 11]`).
3. **Evaluation & Serialization:** Compare all models using **Accuracy**, **F1-Score**, **ROC-AUC**, and **Confusion Matrices**, then export the winning pipeline to `models/best_churn_model.pkl`.

### Phase 5: Dashboard Deployment (`app.py`)
1. **Executive KPI Header:** Dynamically calculates total active policies, overall churn rate (%), total annual premium at risk (RM), and average policy tenure based on sidebar filters.
2. **Interactive Tabs:**
   * *Tab 1 (Business Insights):* Visualizes churn rate by policy type and late payment frequency.
   * *Tab 2 (Live ML Risk Simulator):* Allows underwriters or retention agents to input a policyholder's profile and receive an instant churn probability score with automated retention action recommendations.
   * *Tab 3 (SQL Data Explorer):* Displays live records queried directly from `insurai.db`.

---

## 5. Implementation Timeline & Execution Guidelines

Follow this structured **7-Day Sprint Plan** to execute, document, and publish the project cleanly alongside your IBM coursework:

| Day | Milestone | Key Deliverables & Action Items | Verification Guideline |
| :--- | :--- | :--- | :--- |
| **Day 1** | **Environment & ETL Setup** | Initialize Git repository, create folder structure, write `src/etl_pipeline.py` and `01_Data_Collection_and_Wrangling.ipynb`. | Confirm `data/insurai.db` generates cleanly with 1,500 rows and zero `NULL` values. |
| **Day 2** | **SQL EDA & Business Metrics** | Build `02_SQL_Exploratory_Data_Analysis.ipynb` with at least 5 analytical SQL queries (`GROUP BY`, `CASE WHEN`, subqueries). | Ensure every SQL query includes a markdown cell explaining the business takeaway. |
| **Day 3** | **Statistical & Geospatial Viz** | Build `03_Geospatial_and_Visual_Analytics.ipynb` using `Seaborn` plots and a `Folium` map of Malaysian branches. | Save high-resolution PNG charts into an `assets/` folder for your GitHub README. |
| **Day 4** | **ML Model Benchmarking** | Build `04_ML_Classification_Benchmark.ipynb` and `src/train_model.py` comparing all 4 classifiers via `GridSearchCV`. | Verify `models/best_churn_model.pkl` saves properly and prints a comparison summary table. |
| **Day 5** | **Streamlit Dashboard Build** | Develop `app.py` integrating SQL filtering, KPI metrics, charts, and the live ML prediction form. | Test edge cases locally (`streamlit run app.py`) including empty filter selections. |
| **Day 6** | **Cloud Deployment & README** | Push all code to GitHub, deploy `app.py` to Streamlit Community Cloud, and write a comprehensive `README.md`. | Confirm the public Streamlit URL loads smoothly without dependency errors. |
| **Day 7** | **Resume & LinkedIn Integration** | Update resume project bullets, add the live URL to your LinkedIn Featured section, and link it to your IBM certificate. | Do a 2-minute mock walkthrough explaining the SQL-to-ML pipeline out loud for interviews. |

---

## 6. Best-Practice Guidelines for Maximum Recruiter Impact

1. **Tell a Business Story, Not Just Code:** In every Jupyter Notebook, precede code blocks with a clear **Objective** markdown cell and follow outputs with a **Key Insight** markdown cell (e.g., *"Policyholders with 2+ late payments exhibit a 3x higher churn probability, representing RM X in preventable annual premium loss"*).
2. **Prevent Data Leakage Strictly:** Always keep scaling and one-hot encoding inside a Scikit-Learn `Pipeline` rather than fitting on the entire dataset before `train_test_split`. Interviewers at MNCs (such as AIA, PwC, and PETRONAS) frequently test candidates on this exact concept.
3. **Include Visual Proof in `README.md`:** Embed screenshots of your Streamlit dashboard, your Folium Malaysia map, and your 4-model benchmark comparison table directly at the top of your GitHub `README.md` so recruiters see your work within 5 seconds of clicking your link.
4. **Prepare Your 60-Second Interview Pitch:**
   > *"To apply the IBM Data Science Methodology, I built InsurAI—an end-to-end churn analytics pipeline. I engineered a Python and SQLite ETL pipeline to clean 1,500 policyholder records, ran SQL aggregations and Folium geospatial analysis to pinpoint high-risk segments, benchmarked four classification models using GridSearchCV, and deployed the top model into a live Streamlit dashboard that estimates annual premium at risk."*