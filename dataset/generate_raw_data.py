"""Generate a synthetic, deliberately messy policyholder dataset for InsurAI.

Churn patterns are modelled on what real churn data shows (short tenure,
late payments and unresolved complaints raise churn; long-term policies are
stickier). Each churned customer gets a churn_reason that follows from their
own profile, so the EDA can answer "why did they leave?".

NOTE: this is synthetic data. State that in the README.
Usage: python src/generate_raw_data.py  ->  data/raw_policyholders.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED, N = 42, 1500
rng = np.random.default_rng(SEED)

REGIONS = ["Kuala Lumpur", "Selangor", "Johor", "Penang", "Sarawak"]
REGION_P = [0.28, 0.30, 0.17, 0.13, 0.12]
POLICY_TYPES = ["Medical & Health", "Life Protection", "Motor Comprehensive", "Investment-Linked"]
POLICY_P = [0.30, 0.25, 0.30, 0.15]
BASE_PREMIUM = {"Medical & Health": 210, "Life Protection": 160,
                "Motor Comprehensive": 120, "Investment-Linked": 380}

age = np.clip(rng.normal(41, 11, N).round(), 21, 70).astype(int)
region = rng.choice(REGIONS, N, p=REGION_P)
policy = rng.choice(POLICY_TYPES, N, p=POLICY_P)
tenure = np.clip(rng.gamma(2.2, 14, N).round(), 1, 120).astype(int)
premium = np.array([BASE_PREMIUM[p] for p in policy]) * rng.lognormal(0, 0.25, N)
premium = premium.round(2)
claims = rng.poisson(0.35 + 0.01 * tenure, N).clip(0, 8)
claim_rejected = (claims > 0) & (rng.random(N) < 0.22)
late = rng.poisson(np.where(tenure < 12, 1.3, 0.6), N).clip(0, 6)
tickets = rng.poisson(0.5 + 0.9 * claim_rejected + 0.15 * late, N).clip(0, 8)
competitor_offer = rng.random(N) < 0.18
relocated = rng.random(N) < 0.05

# churn probability (logistic) built from the same drivers
REGION_EFFECT = {"Kuala Lumpur": 0.35, "Selangor": 0.10, "Johor": 0.25,
                 "Penang": -0.30, "Sarawak": -0.45}   # gives the map visible hotspots
z = (-1.95 + np.array([REGION_EFFECT[r] for r in region]) + 1.1 * (tenure < 12) - 0.012 * tenure + 0.55 * late + 0.30 * tickets
     + 0.9 * claim_rejected + 0.7 * competitor_offer + 1.6 * relocated
     + 0.0015 * (premium - premium.mean()) + 0.6 * (policy == "Investment-Linked"))
churned = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)

# churn reason = most plausible cause for THIS customer (weighted random pick)
def pick_reason(i):
    w = {
        "Premium too expensive": 1.0 + 0.004 * max(premium[i] - 150, 0) + 0.4 * late[i],
        "Competitor offered better price": 0.4 + 3.0 * competitor_offer[i],
        "Poor customer service": 0.4 + 0.9 * tickets[i],
        "Claim rejected or dissatisfied": 0.2 + 4.0 * claim_rejected[i],
        "Relocated or moved abroad": 0.2 + 5.0 * relocated[i],
        "No longer needs coverage": 0.8,
    }
    k, p = list(w), np.array(list(w.values()))
    return rng.choice(k, p=p / p.sum())

reason = np.array([pick_reason(i) if churned[i] else None for i in range(N)], dtype=object)

df = pd.DataFrame({
    "policy_id": [f"POL-{10000 + i}" for i in range(N)],
    "age": age, "region": region, "policy_type": policy,
    "tenure_months": tenure, "monthly_premium_rm": premium,
    "claims_count": claims, "late_payments": late.astype(float),
    "support_tickets": tickets, "churned": churned, "churn_reason": reason,
})

# --- inject realistic mess for the ETL to fix -------------------------------
messy = rng.random(N)
df.loc[messy < 0.08, "region"] = df["region"].str.lower()
df.loc[(messy >= 0.08) & (messy < 0.16), "region"] = " " + df["region"] + " "
df.loc[(messy >= 0.16) & (messy < 0.20), "region"] = df["region"].str.upper()
df.loc[rng.random(N) < 0.06, "late_payments"] = np.nan

Path("data").mkdir(exist_ok=True)
df.to_csv("data/raw_policyholders.csv", index=False)
print(f"Wrote {len(df)} rows -> data/raw_policyholders.csv  (churn {df.churned.mean():.1%})")
