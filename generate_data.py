import argparse
import os
import numpy as np
import pandas as pd

def generate_data(rows=5000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, rows)
    tenure_months = rng.integers(1, 73, rows)
    billing_amount = rng.normal(65, 25, rows).clip(10, 180).round(2)
    monthly_usage_hours = rng.gamma(4, 6, rows).clip(0.5, 120).round(2)
    active_days = rng.integers(1, 31, rows)
    login_count = (monthly_usage_hours * rng.uniform(0.4, 1.4, rows)).astype(int).clip(0, 200)
    avg_session_min = rng.normal(18, 8, rows).clip(2, 90).round(2)
    support_tickets = rng.poisson(0.8, rows)
    sla_breaches = rng.binomial(3, 0.08, rows)
    nps_score = rng.integers(0, 11, rows)
    last_payment_days_ago = rng.integers(0, 35, rows)
    last_campaign_days_ago = rng.integers(1, 90, rows)
    email_opens = rng.poisson(3, rows)
    email_clicks = np.minimum(email_opens, rng.poisson(1, rows))
    plan_tier = rng.choice(["Basic", "Standard", "Premium"], rows, p=[0.45, 0.38, 0.17])
    region = rng.choice(["North", "South", "East", "West"], rows)
    is_autopay = rng.choice([0, 1], rows, p=[0.45, 0.55])
    is_discounted = rng.choice([0, 1], rows, p=[0.70, 0.30])
    has_family_bundle = rng.choice([0, 1], rows, p=[0.65, 0.35])

    engagement_rate = active_days / 30
    support_intensity = support_tickets + 3 * sla_breaches
    price_to_tenure = billing_amount / (tenure_months + 1)

    z = (
        -2.5
        + 1.8 * (engagement_rate < 0.35)
        + 1.2 * (monthly_usage_hours < 12)
        + 0.08 * support_intensity
        + 0.10 * last_payment_days_ago
        - 0.09 * tenure_months
        - 0.18 * nps_score
        + 0.015 * billing_amount
        - 0.60 * is_autopay
        - 0.35 * has_family_bundle
        + 0.45 * (plan_tier == "Basic")
        + 0.35 * (price_to_tenure > 12)
    )

    p = 1 / (1 + np.exp(-z))
    churned_next_cycle = rng.binomial(1, p)

    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, rows + 1)],
        "age": age,
        "tenure_months": tenure_months,
        "billing_amount": billing_amount,
        "monthly_usage_hours": monthly_usage_hours,
        "active_days": active_days,
        "login_count": login_count,
        "avg_session_min": avg_session_min,
        "support_tickets": support_tickets,
        "sla_breaches": sla_breaches,
        "nps_score": nps_score,
        "last_payment_days_ago": last_payment_days_ago,
        "last_campaign_days_ago": last_campaign_days_ago,
        "email_opens": email_opens,
        "email_clicks": email_clicks,
        "plan_tier": plan_tier,
        "region": region,
        "is_autopay": is_autopay,
        "is_discounted": is_discounted,
        "has_family_bundle": has_family_bundle,
        "churned_next_cycle": churned_next_cycle,
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    df = generate_data(args.rows)
    df.to_csv("data/churn_frame.csv", index=False)
    print("Saved data/churn_frame.csv")
    print("Rows:", len(df))
    print("Churn rate:", round(df["churned_next_cycle"].mean(), 4))

if __name__ == "__main__":
    main()
