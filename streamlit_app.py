import json
import os
import subprocess
import sys
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction Dashboard", page_icon="📉", layout="wide")

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def ensure_synthetic_ready():
    if not os.path.exists("data/churn_frame.csv"):
        run_command(f"{sys.executable} src/generate_data.py --rows 5000")
    if not os.path.exists("models/churn_model.joblib"):
        run_command(f"{sys.executable} src/train_model.py")

def read_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

st.title("📉 Customer Churn Prediction Dashboard")
st.caption("Industry-style churn scoring, risk segmentation, retention actions, and model performance monitoring.")

tab1, tab2, tab3 = st.tabs(["Synthetic Churn Dashboard", "IBM Telco Dataset", "Single Customer Prediction"])

with tab1:
    ensure_synthetic_ready()
    df = pd.read_csv("data/churn_frame.csv")
    metrics = read_json("outputs/metrics.json")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{df['churned_next_cycle'].mean()*100:.1f}%")
    c3.metric("ROC-AUC", metrics.get("roc_auc", "NA"))
    c4.metric("Lift@10%", metrics.get("lift_at_10_percent", "NA"))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(df, x="tenure_months", color="churned_next_cycle", title="Tenure vs Churn"), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(df, x="monthly_usage_hours", color="churned_next_cycle", title="Usage vs Churn"), use_container_width=True)

    st.subheader("Top 50 Churn Watchlist")
    if os.path.exists("outputs/top_50_churn_watchlist.csv"):
        st.dataframe(pd.read_csv("outputs/top_50_churn_watchlist.csv"), use_container_width=True)

    st.subheader("Model Charts")
    cols = st.columns(4)
    for col, img, title in zip(cols, ["images/confusion_matrix.png", "images/roc_curve.png", "images/pr_curve.png", "images/feature_importance.png"], ["Confusion Matrix", "ROC Curve", "PR Curve", "Feature Importance"]):
        with col:
            st.write(title)
            if os.path.exists(img):
                st.image(img, use_container_width=True)

with tab2:
    st.subheader("IBM Telco Customer Churn Dataset")
    if os.path.exists("data/telco_customer_churn.csv"):
        telco = pd.read_csv("data/telco_customer_churn.csv")
        st.success("Dataset found: data/telco_customer_churn.csv")
        st.dataframe(telco.head(20), use_container_width=True)

        if st.button("Train IBM Telco Model"):
            with st.spinner("Training model..."):
                out, err, code = run_command(f"{sys.executable} src/train_telco_model.py")
                if code == 0:
                    st.success("IBM Telco model trained successfully.")
                    st.code(out)
                else:
                    st.error("Training failed.")
                    st.code(err)

        metrics = read_json("outputs/telco_metrics.json")
        if metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{metrics.get('rows', 0):,}")
            c2.metric("Churn Rate", f"{metrics.get('churn_rate', 0)*100:.1f}%")
            c3.metric("ROC-AUC", metrics.get("roc_auc", "NA"))
            c4.metric("Lift@10%", metrics.get("lift_at_10_percent", "NA"))

        if os.path.exists("outputs/telco_top_50_churn_watchlist.csv"):
            st.subheader("IBM Telco Top 50 Churn Watchlist")
            st.dataframe(pd.read_csv("outputs/telco_top_50_churn_watchlist.csv"), use_container_width=True)

        cols = st.columns(4)
        for col, img, title in zip(cols, ["images/telco_confusion_matrix.png", "images/telco_roc_curve.png", "images/telco_pr_curve.png", "images/telco_feature_importance.png"], ["Confusion Matrix", "ROC Curve", "PR Curve", "Feature Importance"]):
            with col:
                st.write(title)
                if os.path.exists(img):
                    st.image(img, use_container_width=True)
    else:
        st.warning("Upload IBM Telco dataset to GitHub at: data/telco_customer_churn.csv")

with tab3:
    ensure_synthetic_ready()
    model = joblib.load("models/churn_model.joblib")

    st.subheader("Predict One Customer")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 80, 35)
        tenure_months = st.slider("Tenure Months", 1, 72, 12)
        billing_amount = st.number_input("Billing Amount", 10.0, 200.0, 75.0)
        monthly_usage_hours = st.number_input("Monthly Usage Hours", 0.0, 150.0, 10.0)
        active_days = st.slider("Active Days", 1, 30, 8)
    with col2:
        login_count = st.number_input("Login Count", 0, 250, 15)
        avg_session_min = st.number_input("Average Session Minutes", 1.0, 100.0, 15.0)
        support_tickets = st.slider("Support Tickets", 0, 10, 2)
        sla_breaches = st.slider("SLA Breaches", 0, 5, 1)
        nps_score = st.slider("NPS Score", 0, 10, 4)
    with col3:
        last_payment_days_ago = st.slider("Last Payment Days Ago", 0, 35, 20)
        last_campaign_days_ago = st.slider("Last Campaign Days Ago", 1, 90, 30)
        email_opens = st.slider("Email Opens", 0, 20, 2)
        email_clicks = st.slider("Email Clicks", 0, 20, 0)
        plan_tier = st.selectbox("Plan Tier", ["Basic", "Standard", "Premium"])
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        is_autopay = st.selectbox("Autopay", [0, 1])
        is_discounted = st.selectbox("Discounted", [0, 1])
        has_family_bundle = st.selectbox("Family Bundle", [0, 1])

    row = pd.DataFrame([{
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
    }])

    row["engagement_rate"] = (row["active_days"] / 30).clip(0, 1)
    row["usage_per_login"] = row["monthly_usage_hours"] / (row["login_count"] + 0.001)
    row["support_intensity"] = row["support_tickets"] + 3 * row["sla_breaches"]
    row["email_ctr"] = row["email_clicks"] / (row["email_opens"] + 0.001)
    row["price_to_tenure"] = row["billing_amount"] / (row["tenure_months"] + 0.001)

    if st.button("Predict Churn"):
        prob = float(model.predict_proba(row)[0, 1])
        if prob >= 0.50:
            risk = "High Risk"
            action = "Priority retention call + personalized discount"
        elif prob >= 0.25:
            risk = "Medium Risk"
            action = "Personalized engagement campaign"
        else:
            risk = "Low Risk"
            action = "Regular engagement email"

        st.metric("Churn Probability", f"{prob * 100:.1f}%")
        st.metric("Risk Segment", risk)
        st.success(f"Recommended Action: {action}")
