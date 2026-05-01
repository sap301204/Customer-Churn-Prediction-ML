import pandas as pd

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df

def add_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engagement_rate"] = (df["active_days"] / 30).clip(0, 1)
    df["usage_per_login"] = df["monthly_usage_hours"] / (df["login_count"] + 0.001)
    df["support_intensity"] = df["support_tickets"] + 3 * df["sla_breaches"]
    df["email_ctr"] = df["email_clicks"] / (df["email_opens"] + 0.001)
    df["price_to_tenure"] = df["billing_amount"] / (df["tenure_months"] + 0.001)
    return df

def add_telco_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["total_charges", "monthly_charges", "tenure", "cltv", "churn_score", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "monthly_charges" in df.columns and "tenure" in df.columns:
        df["monthly_charge_to_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)
    if "total_charges" in df.columns and "tenure" in df.columns:
        df["avg_total_charge_per_month"] = df["total_charges"] / (df["tenure"] + 1)
    if "contract" in df.columns:
        df["is_month_to_month"] = df["contract"].astype(str).str.lower().str.contains("month", na=False).astype(int)
    if "online_security" in df.columns and "tech_support" in df.columns:
        df["security_support_gap"] = (
            (df["online_security"].astype(str).str.lower() == "no").astype(int) +
            (df["tech_support"].astype(str).str.lower() == "no").astype(int)
        )
    return df

def recommend_action(probability: float, row=None) -> str:
    row = row or {}
    if probability >= 0.75:
        return "Priority retention call + personalized discount"
    if probability >= 0.50:
        return "Targeted retention offer"
    if probability >= 0.25:
        return "Personalized engagement campaign"
    return "Regular engagement email"

def risk_segment(probability: float) -> str:
    if probability >= 0.50:
        return "High Risk"
    if probability >= 0.25:
        return "Medium Risk"
    return "Low Risk"
