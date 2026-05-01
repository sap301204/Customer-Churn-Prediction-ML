import json
import os
import subprocess
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide"
)


# -----------------------------
# PATH CONFIG
# -----------------------------
SYNTHETIC_DATA_PATH = "data/churn_frame.csv"
TELCO_CSV_PATH = "data/telco_customer_churn.csv"
TELCO_XLSX_PATH = "Telco_customer_churn.xlsx"

SYNTHETIC_MODEL_PATH = "models/churn_model.joblib"
TELCO_MODEL_PATH = "models/telco_churn_model.joblib"

METRICS_PATH = "outputs/metrics.json"
TELCO_METRICS_PATH = "outputs/telco_metrics.json"

WATCHLIST_PATH = "outputs/top_50_churn_watchlist.csv"
TELCO_WATCHLIST_PATH = "outputs/telco_top_50_churn_watchlist.csv"


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def run_command(command):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr, result.returncode


def ensure_folders():
    folders = ["data", "models", "outputs", "images", "src"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def read_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def convert_telco_excel_to_csv_if_needed():
    """
    Your GitHub repo currently appears to have Telco_customer_churn.xlsx.
    This function converts it to data/telco_customer_churn.csv automatically.
    """
    if os.path.exists(TELCO_CSV_PATH):
        return True

    if os.path.exists(TELCO_XLSX_PATH):
        try:
            telco_df = pd.read_excel(TELCO_XLSX_PATH)
            ensure_folders()
            telco_df.to_csv(TELCO_CSV_PATH, index=False)
            return True
        except Exception as e:
            st.error("Found Excel file, but could not convert it to CSV.")
            st.code(str(e))
            return False

    return False


def ensure_synthetic_ready():
    """
    This fixes your main error.
    Earlier code checked churn_frame.csv in root,
    but the app reads data/churn_frame.csv.
    """
    ensure_folders()

    if not os.path.exists(SYNTHETIC_DATA_PATH):
        if os.path.exists("generate_data.py"):
            out, err, code = run_command(f"{sys.executable} generate_data.py --rows 5000")
        elif os.path.exists("src/generate_data.py"):
            out, err, code = run_command(f"{sys.executable} src/generate_data.py --rows 5000")
        else:
            st.error("generate_data.py not found. Please upload it to GitHub.")
            return False

        if code != 0:
            st.error("Failed to generate synthetic churn data.")
            st.code(err)
            return False

        # If script created file in root, move it to data/
        if os.path.exists("churn_frame.csv") and not os.path.exists(SYNTHETIC_DATA_PATH):
            pd.read_csv("churn_frame.csv").to_csv(SYNTHETIC_DATA_PATH, index=False)

    if not os.path.exists(SYNTHETIC_MODEL_PATH):
        if os.path.exists("train_model.py"):
            out, err, code = run_command(f"{sys.executable} train_model.py")
        elif os.path.exists("src/train_model.py"):
            out, err, code = run_command(f"{sys.executable} src/train_model.py")
        else:
            st.warning("train_model.py not found. Dashboard will show data only.")
            return os.path.exists(SYNTHETIC_DATA_PATH)

        if code != 0:
            st.warning("Model training failed, but dashboard can still show dataset charts.")
            st.code(err)

    return os.path.exists(SYNTHETIC_DATA_PATH)


def risk_action(prob):
    if prob >= 0.50:
        return "High Risk", "Priority retention call + personalized discount"
    elif prob >= 0.25:
        return "Medium Risk", "Personalized engagement campaign"
    else:
        return "Low Risk", "Regular engagement email"


# -----------------------------
# DASHBOARD HEADER
# -----------------------------
st.title("📉 Customer Churn Prediction Dashboard")
st.caption(
    "Industry-style churn scoring, risk segmentation, retention actions, "
    "and model performance monitoring."
)

tab1, tab2, tab3 = st.tabs([
    "Synthetic Churn Dashboard",
    "IBM Telco Dataset",
    "Single Customer Prediction"
])


# -----------------------------
# TAB 1: SYNTHETIC DASHBOARD
# -----------------------------
with tab1:
    st.subheader("Synthetic Customer Churn Dashboard")

    ready = ensure_synthetic_ready()

    if not ready:
        st.error("Synthetic dataset is not available. Please upload or generate data/churn_frame.csv.")
        st.stop()

    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    metrics = read_json(METRICS_PATH)

    if "churned_next_cycle" not in df.columns:
        st.error("Column 'churned_next_cycle' not found in data/churn_frame.csv.")
        st.dataframe(df.head())
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{df['churned_next_cycle'].mean() * 100:.1f}%")
    c3.metric("ROC-AUC", metrics.get("roc_auc", "NA"))
    c4.metric("Lift@10%", metrics.get("lift_at_10_percent", "NA"))

    st.divider()

    left, right = st.columns(2)

    with left:
        if "tenure_months" in df.columns:
            fig = px.histogram(
                df,
                x="tenure_months",
                color="churned_next_cycle",
                title="Tenure vs Churn"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("tenure_months column not found.")

    with right:
        if "monthly_usage_hours" in df.columns:
            fig = px.histogram(
                df,
                x="monthly_usage_hours",
                color="churned_next_cycle",
                title="Monthly Usage Hours vs Churn"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("monthly_usage_hours column not found.")

    st.subheader("Top 50 Churn Watchlist")

    if os.path.exists(WATCHLIST_PATH):
        st.dataframe(pd.read_csv(WATCHLIST_PATH), use_container_width=True)
    else:
        st.info("Watchlist not found yet. Train the model to generate outputs/top_50_churn_watchlist.csv.")

    st.subheader("Model Charts")

    chart_files = [
        ("images/confusion_matrix.png", "Confusion Matrix"),
        ("images/roc_curve.png", "ROC Curve"),
        ("images/pr_curve.png", "Precision-Recall Curve"),
        ("images/feature_importance.png", "Feature Importance")
    ]

    cols = st.columns(4)

    for col, (img, title) in zip(cols, chart_files):
        with col:
            st.write(title)
            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.info("Not available")


# -----------------------------
# TAB 2: IBM TELCO DATASET
# -----------------------------
with tab2:
    st.subheader("IBM Telco Customer Churn Dataset")

    telco_ready = convert_telco_excel_to_csv_if_needed()

    if telco_ready:
        telco = pd.read_csv(TELCO_CSV_PATH)

        st.success("Dataset found successfully.")
        st.caption(f"Using: {TELCO_CSV_PATH}")
        st.dataframe(telco.head(20), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(telco):,}")
        c2.metric("Columns", f"{telco.shape[1]:,}")

        possible_churn_cols = [
            "Churn", "churn", "Customer Status", "customer_status",
            "Churn Label", "churn_label"
        ]

        churn_col = None
        for col in possible_churn_cols:
            if col in telco.columns:
                churn_col = col
                break

        if churn_col:
            churn_rate = telco[churn_col].astype(str).str.lower().str.contains("yes|churn|1|true").mean()
            c3.metric("Approx Churn Rate", f"{churn_rate * 100:.1f}%")
        else:
            c3.metric("Target Column", "Not detected")

        st.divider()

        if st.button("Train IBM Telco Model"):
            with st.spinner("Training IBM Telco model..."):
                if os.path.exists("src/train_telco_model.py"):
                    out, err, code = run_command(f"{sys.executable} src/train_telco_model.py")
                elif os.path.exists("train_telco_model.py"):
                    out, err, code = run_command(f"{sys.executable} train_telco_model.py")
                else:
                    out, err, code = "", "train_telco_model.py not found.", 1

                if code == 0:
                    st.success("IBM Telco model trained successfully.")
                    st.code(out)
                else:
                    st.error("Training failed.")
                    st.code(err)

        metrics = read_json(TELCO_METRICS_PATH)

        if metrics:
            st.subheader("IBM Telco Model Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", f"{metrics.get('rows', 0):,}")
            m2.metric("Accuracy", metrics.get("accuracy", "NA"))
            m3.metric("ROC-AUC", metrics.get("roc_auc", "NA"))
            m4.metric("PR-AUC", metrics.get("pr_auc", "NA"))

        if os.path.exists(TELCO_WATCHLIST_PATH):
            st.subheader("IBM Telco Top 50 Churn Watchlist")
            st.dataframe(pd.read_csv(TELCO_WATCHLIST_PATH), use_container_width=True)

        st.subheader("IBM Telco Model Charts")

        telco_chart_files = [
            ("images/telco_confusion_matrix.png", "Confusion Matrix"),
            ("images/telco_roc_curve.png", "ROC Curve"),
            ("images/telco_pr_curve.png", "Precision-Recall Curve"),
            ("images/telco_feature_importance.png", "Feature Importance")
        ]

        cols = st.columns(4)

        for col, (img, title) in zip(cols, telco_chart_files):
            with col:
                st.write(title)
                if os.path.exists(img):
                    st.image(img, use_container_width=True)
                else:
                    st.info("Not available")

    else:
        st.warning(
            "IBM Telco dataset not found. Upload it as either "
            "`data/telco_customer_churn.csv` or `Telco_customer_churn.xlsx`."
        )


# -----------------------------
# TAB 3: SINGLE CUSTOMER PREDICTION
# -----------------------------
with tab3:
    st.subheader("Predict One Customer")

    ready = ensure_synthetic_ready()

    if not ready:
        st.error("Synthetic model/data is not ready. Please check your files.")
        st.stop()

    if not os.path.exists(SYNTHETIC_MODEL_PATH):
        st.error("Model file not found: models/churn_model.joblib")
        st.info("Train the model first or upload the model file.")
        st.stop()

    model = joblib.load(SYNTHETIC_MODEL_PATH)

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
        risk, action = risk_action(prob)

        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob * 100:.1f}%")
        r2.metric("Risk Segment", risk)
        r3.metric("Recommended Action", action)

        if risk == "High Risk":
            st.error("This customer needs immediate retention attention.")
        elif risk == "Medium Risk":
            st.warning("This customer should receive a targeted engagement campaign.")
        else:
            st.success("This customer is currently low risk.")
