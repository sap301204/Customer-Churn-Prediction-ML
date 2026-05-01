import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds business-friendly engineered features for churn prediction.
    Works with the synthetic churn dataset.
    """

    df = df.copy()

    # Engagement rate: how active the customer is in a month
    if "active_days" in df.columns:
        df["engagement_rate"] = (df["active_days"] / 30.0).clip(0, 1)
    else:
        df["engagement_rate"] = 0

    # Usage per login: average usage intensity
    if "monthly_usage_hours" in df.columns and "login_count" in df.columns:
        df["usage_per_login"] = df["monthly_usage_hours"] / (df["login_count"] + 0.001)
    else:
        df["usage_per_login"] = 0

    # Support intensity: support tickets + SLA breaches impact
    if "support_tickets" in df.columns and "sla_breaches" in df.columns:
        df["support_intensity"] = df["support_tickets"] + 3 * df["sla_breaches"]
    else:
        df["support_intensity"] = 0

    # Email click-through rate
    if "email_clicks" in df.columns and "email_opens" in df.columns:
        df["email_ctr"] = df["email_clicks"] / (df["email_opens"] + 0.001)
    else:
        df["email_ctr"] = 0

    # Price pressure compared to tenure
    if "billing_amount" in df.columns and "tenure_months" in df.columns:
        df["price_to_tenure"] = df["billing_amount"] / (df["tenure_months"] + 0.001)
    else:
        df["price_to_tenure"] = 0

    return df
