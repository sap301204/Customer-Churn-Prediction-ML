import os
import sys
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from features import add_synthetic_features, recommend_action, risk_segment

app = FastAPI(title="Customer Churn Prediction API")

class Customer(BaseModel):
    age: int = 35
    tenure_months: int = 12
    billing_amount: float = 75
    monthly_usage_hours: float = 10
    active_days: int = 8
    login_count: int = 15
    avg_session_min: float = 15
    support_tickets: int = 2
    sla_breaches: int = 1
    nps_score: int = 4
    last_payment_days_ago: int = 20
    last_campaign_days_ago: int = 30
    email_opens: int = 2
    email_clicks: int = 0
    plan_tier: str = "Basic"
    region: str = "West"
    is_autopay: int = 0
    is_discounted: int = 0
    has_family_bundle: int = 0

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/score")
def score(customer: Customer) -> Dict[str, Any]:
    if not os.path.exists("models/churn_model.joblib"):
        return {"error": "Model not found. Run python src/train_model.py first."}

    model = joblib.load("models/churn_model.joblib")
    row = pd.DataFrame([customer.model_dump()])
    row = add_synthetic_features(row)
    prob = float(model.predict_proba(row)[0, 1])
    return {
        "churn_probability": round(prob, 4),
        "risk_segment": risk_segment(prob),
        "recommended_action": recommend_action(prob)
    }
