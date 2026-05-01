from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Numeric columns used by synthetic churn dataset
NUM = [
    "age",
    "tenure_months",
    "billing_amount",
    "monthly_usage_hours",
    "active_days",
    "login_count",
    "avg_session_min",
    "support_tickets",
    "sla_breaches",
    "nps_score",
    "last_payment_days_ago",
    "last_campaign_days_ago",
    "email_opens",
    "email_clicks",
    "engagement_rate",
    "usage_per_login",
    "support_intensity",
    "email_ctr",
    "price_to_tenure",
]


# Categorical columns used by synthetic churn dataset
CAT = [
    "plan_tier",
    "region",
    "is_autopay",
    "is_discounted",
    "has_family_bundle",
]


numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


pre = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUM),
        ("cat", categorical_pipeline, CAT),
    ],
    remainder="drop"
)
