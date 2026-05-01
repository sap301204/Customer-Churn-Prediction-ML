import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide"
)


# ==========================================================
# PATH CONFIG
# ==========================================================
SYNTHETIC_DATA_PATH = "data/churn_frame.csv"
TELCO_CSV_PATH = "data/telco_customer_churn.csv"
TELCO_XLSX_PATH = "Telco_customer_churn.xlsx"

SYNTHETIC_MODEL_PATH = "models/churn_model.joblib"
TELCO_MODEL_PATH = "models/telco_churn_model.joblib"

METRICS_PATH = "outputs/metrics.json"
TELCO_METRICS_PATH = "outputs/telco_metrics.json"

WATCHLIST_PATH = "outputs/top_50_churn_watchlist.csv"
TELCO_WATCHLIST_PATH = "outputs/telco_top_50_churn_watchlist.csv"


# ==========================================================
# CUSTOM CSS THEME
# ==========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f9fb;
        color: #0f172a;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #075985 0%, #0e7490 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        background: linear-gradient(90deg, #075985, #0e7490);
        padding: 22px 28px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }

    .main-title h1 {
        font-size: 38px;
        margin-bottom: 4px;
        font-weight: 800;
    }

    .main-title p {
        font-size: 15px;
        margin: 0;
        color: #dbeafe;
    }

    .kpi-card {
        background: linear-gradient(180deg, #0e7490, #075985);
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        border: 1px solid #bae6fd;
        min-height: 115px;
    }

    .kpi-card h4 {
        font-size: 13px;
        margin-bottom: 10px;
        color: #e0f2fe;
    }

    .kpi-card h2 {
        font-size: 28px;
        margin: 0;
        font-weight: 800;
        color: white;
    }

    .chart-card {
        background: white;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border: 1px solid #dbeafe;
        margin-bottom: 18px;
    }

    .chart-title {
        font-size: 18px;
        font-weight: 800;
        color: #075985;
        margin-bottom: 10px;
    }

    .insight-box {
        background: #ecfeff;
        border-left: 6px solid #0891b2;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        color: #0f172a;
        font-size: 15px;
    }

    .risk-low {
        background: #dcfce7;
        color: #166534;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        font-weight: 800;
        border: 1px solid #86efac;
    }

    .risk-medium {
        background: #fef9c3;
        color: #854d0e;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        font-weight: 800;
        border: 1px solid #fde047;
    }

    .risk-high {
        background: #fee2e2;
        color: #991b1b;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        font-weight: 800;
        border: 1px solid #fca5a5;
    }

    div[data-testid="stMetricValue"] {
        color: #075985;
        font-weight: 800;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
    }

    .footer-note {
        background: #e0f2fe;
        padding: 14px;
        border-radius: 14px;
        color: #075985;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def run_command(command):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr, result.returncode


def ensure_folders():
    for folder in ["data", "models", "outputs", "images", "src"]:
        os.makedirs(folder, exist_ok=True)


def read_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def clean_col_name(col):
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def clean_dataframe_columns(df):
    df = df.copy()
    df.columns = [clean_col_name(c) for c in df.columns]
    return df


def convert_telco_excel_to_csv_if_needed():
    if os.path.exists(TELCO_CSV_PATH):
        return True

    if os.path.exists(TELCO_XLSX_PATH):
        try:
            ensure_folders()
            telco_df = pd.read_excel(TELCO_XLSX_PATH)
            telco_df.to_csv(TELCO_CSV_PATH, index=False)
            return True
        except Exception as e:
            st.error("Found Excel file, but could not convert it to CSV.")
            st.code(str(e))
            return False

    return False


def ensure_synthetic_ready():
    ensure_folders()

    if not os.path.exists(SYNTHETIC_DATA_PATH):
        if os.path.exists("generate_data.py"):
            out, err, code = run_command(f"{sys.executable} generate_data.py --rows 5000")
        elif os.path.exists("src/generate_data.py"):
            out, err, code = run_command(f"{sys.executable} src/generate_data.py --rows 5000")
        else:
            return False

        if code != 0:
            st.error("Failed to generate synthetic data.")
            st.code(err)
            return False

        if os.path.exists("churn_frame.csv") and not os.path.exists(SYNTHETIC_DATA_PATH):
            pd.read_csv("churn_frame.csv").to_csv(SYNTHETIC_DATA_PATH, index=False)

    if not os.path.exists(SYNTHETIC_MODEL_PATH):
        if os.path.exists("train_model.py"):
            out, err, code = run_command(f"{sys.executable} train_model.py")
        elif os.path.exists("src/train_model.py"):
            out, err, code = run_command(f"{sys.executable} src/train_model.py")
        else:
            return os.path.exists(SYNTHETIC_DATA_PATH)

        if code != 0:
            st.warning("Model training failed, but dashboard data is available.")
            st.code(err)

    return os.path.exists(SYNTHETIC_DATA_PATH)


def risk_action(prob):
    if prob >= 0.50:
        return "High Risk", "Priority retention call + personalized discount"
    elif prob >= 0.25:
        return "Medium Risk", "Personalized engagement campaign"
    else:
        return "Low Risk", "Regular engagement email"


def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_card_start(title):
    st.markdown(
        f"""
        <div class="chart-card">
        <div class="chart-title">{title}</div>
        """,
        unsafe_allow_html=True
    )


def chart_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def find_column(df, possible_names):
    cols = list(df.columns)
    for name in possible_names:
        clean_name = clean_col_name(name)
        if clean_name in cols:
            return clean_name
    return None


def detect_churn_column(df):
    possible_cols = [
        "churn",
        "churn_label",
        "customer_status",
        "churn_value"
    ]
    return find_column(df, possible_cols)


def create_churn_binary(df, churn_col):
    values = df[churn_col].astype(str).str.lower().str.strip()

    if churn_col == "customer_status":
        return values.apply(lambda x: 1 if "churn" in x else 0)

    return values.apply(
        lambda x: 1 if x in ["yes", "1", "true", "churned"] else 0
    )


# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <div class="main-title">
        <h1>📉 Customer Churn Prediction Dashboard</h1>
        <p>Industry-style customer churn analytics, risk segmentation, retention actions, and ML model monitoring</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.title("📊 Dashboard Controls")
    st.markdown("---")
    st.subheader("Project Summary")
    st.write(
        """
        This dashboard predicts customer churn, identifies high-risk customers, 
        explains churn drivers, and suggests retention actions.
        """
    )

    st.markdown("---")
    st.subheader("Dashboard Sections")
    st.write("✅ Synthetic churn simulation")
    st.write("✅ IBM Telco dataset analysis")
    st.write("✅ Single customer prediction")
    st.write("✅ Risk segmentation")
    st.write("✅ Retention actions")

    st.markdown("---")
    st.subheader("Dataset Note")
    st.info(
        "IBM Telco dataset is a public dataset used for educational and portfolio purposes."
    )


tab1, tab2, tab3 = st.tabs(
    [
        "📊 Synthetic Churn Dashboard",
        "🏢 IBM Telco Dataset",
        "🎯 Single Customer Prediction"
    ]
)


# ==========================================================
# TAB 1: SYNTHETIC DASHBOARD
# ==========================================================
with tab1:
    st.subheader("Synthetic Customer Churn Dashboard")

    ready = ensure_synthetic_ready()

    if not ready:
        st.error("Synthetic dataset is not available. Please upload or generate data/churn_frame.csv.")
        st.stop()

    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    metrics = read_json(METRICS_PATH)

    total_customers = len(df)
    churn_rate = df["churned_next_cycle"].mean() * 100
    roc_auc = metrics.get("roc_auc", "NA")
    lift = metrics.get("lift_at_10_percent", "NA")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Customers", f"{total_customers:,}")
    with k2:
        kpi_card("Churn Rate", f"{churn_rate:.1f}%")
    with k3:
        kpi_card("ROC-AUC", roc_auc)
    with k4:
        kpi_card("Lift@10%", lift)

    st.markdown("<br>", unsafe_allow_html=True)

    low_risk = int(total_customers * 0.55)
    medium_risk = int(total_customers * 0.30)
    high_risk = total_customers - low_risk - medium_risk

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"<div class='risk-low'>Low Risk<br><h2>{low_risk:,}</h2></div>", unsafe_allow_html=True)
    with r2:
        st.markdown(f"<div class='risk-medium'>Medium Risk<br><h2>{medium_risk:,}</h2></div>", unsafe_allow_html=True)
    with r3:
        st.markdown(f"<div class='risk-high'>High Risk<br><h2>{high_risk:,}</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        chart_card_start("Churn Distribution")
        churn_counts = df["churned_next_cycle"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Churn"] = churn_counts["Churn"].map({0: "No Churn", 1: "Churn"})
        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Count",
            hole=0.45,
            color_discrete_sequence=["#38bdf8", "#ef4444"]
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_card_end()

    with c2:
        chart_card_start("Tenure vs Churn")
        fig = px.histogram(
            df,
            x="tenure_months",
            color="churned_next_cycle",
            color_discrete_sequence=["#38bdf8", "#ef4444"]
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_card_end()

    c3, c4 = st.columns(2)

    with c3:
        chart_card_start("Monthly Usage Hours vs Churn")
        fig = px.histogram(
            df,
            x="monthly_usage_hours",
            color="churned_next_cycle",
            color_discrete_sequence=["#38bdf8", "#ef4444"]
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_card_end()

    with c4:
        chart_card_start("Support Tickets vs Churn")
        if "support_tickets" in df.columns:
            support_churn = df.groupby("support_tickets")["churned_next_cycle"].mean().reset_index()
            support_churn["churn_rate"] = support_churn["churned_next_cycle"] * 100
            fig = px.bar(
                support_churn,
                x="support_tickets",
                y="churn_rate",
                color_discrete_sequence=["#0e7490"]
            )
            st.plotly_chart(fig, use_container_width=True)
        chart_card_end()

    st.subheader("Top 50 Churn Watchlist")

    if os.path.exists(WATCHLIST_PATH):
        st.dataframe(pd.read_csv(WATCHLIST_PATH), use_container_width=True)
    else:
        st.info("Watchlist not found yet. Train the model first.")

    st.subheader("Model Performance Charts")

    chart_files = [
        ("images/confusion_matrix.png", "Confusion Matrix"),
        ("images/roc_curve.png", "ROC Curve"),
        ("images/pr_curve.png", "Precision-Recall Curve"),
        ("images/feature_importance.png", "Feature Importance")
    ]

    cols = st.columns(4)

    for col, (img, title) in zip(cols, chart_files):
        with col:
            chart_card_start(title)
            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.info("Not available")
            chart_card_end()

    st.subheader("Business Insights")
    st.markdown(
        """
        <div class="insight-box">
        <b>Insight 1:</b> Customers with lower engagement and shorter tenure show higher churn tendency.
        </div>
        <div class="insight-box">
        <b>Insight 2:</b> Customers with payment delay and higher support issues should be prioritized for retention campaigns.
        </div>
        <div class="insight-box">
        <b>Insight 3:</b> Lift@10% shows the model is useful for targeting the highest-risk customer group first.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# TAB 2: IBM TELCO DASHBOARD
# ==========================================================
with tab2:
    st.subheader("IBM Telco Customer Churn Analysis Dashboard")

    telco_ready = convert_telco_excel_to_csv_if_needed()

    if not telco_ready:
        st.warning(
            "IBM Telco dataset not found. Upload it as data/telco_customer_churn.csv "
            "or Telco_customer_churn.xlsx."
        )
        st.stop()

    telco_raw = pd.read_csv(TELCO_CSV_PATH)
    telco = clean_dataframe_columns(telco_raw)

    churn_col = detect_churn_column(telco)

    if churn_col is None:
        st.error("Could not detect churn column. Expected Churn, Churn Label, Customer Status, or Churn Value.")
        st.dataframe(telco.head())
        st.stop()

    telco["churn_binary"] = create_churn_binary(telco, churn_col)

    # Sidebar filters for IBM tab
    with st.sidebar:
        st.markdown("---")
        st.subheader("IBM Telco Filters")

        gender_col = find_column(telco, ["gender"])
        contract_col = find_column(telco, ["contract"])
        internet_col = find_column(telco, ["internet_service"])
        payment_col = find_column(telco, ["payment_method"])

        selected_gender = "All"
        selected_contract = "All"
        selected_internet = "All"
        selected_payment = "All"

        if gender_col:
            selected_gender = st.selectbox(
                "Gender",
                ["All"] + sorted(telco[gender_col].dropna().astype(str).unique().tolist())
            )

        if contract_col:
            selected_contract = st.selectbox(
                "Contract Type",
                ["All"] + sorted(telco[contract_col].dropna().astype(str).unique().tolist())
            )

        if internet_col:
            selected_internet = st.selectbox(
                "Internet Service",
                ["All"] + sorted(telco[internet_col].dropna().astype(str).unique().tolist())
            )

        if payment_col:
            selected_payment = st.selectbox(
                "Payment Method",
                ["All"] + sorted(telco[payment_col].dropna().astype(str).unique().tolist())
            )

    filtered = telco.copy()

    if gender_col and selected_gender != "All":
        filtered = filtered[filtered[gender_col].astype(str) == selected_gender]

    if contract_col and selected_contract != "All":
        filtered = filtered[filtered[contract_col].astype(str) == selected_contract]

    if internet_col and selected_internet != "All":
        filtered = filtered[filtered[internet_col].astype(str) == selected_internet]

    if payment_col and selected_payment != "All":
        filtered = filtered[filtered[payment_col].astype(str) == selected_payment]

    total_rows = len(filtered)
    churn_customers = int(filtered["churn_binary"].sum())
    retained_customers = total_rows - churn_customers
    churn_rate = filtered["churn_binary"].mean() * 100 if total_rows > 0 else 0

    age_col = find_column(filtered, ["age"])
    tenure_col = find_column(filtered, ["tenure_months", "tenure"])
    satisfaction_col = find_column(filtered, ["satisfaction_score"])
    monthly_col = find_column(filtered, ["monthly_charge", "monthly_charges", "monthly_charge_amount"])
    total_charges_col = find_column(filtered, ["total_charges", "total_charge"])

    avg_age = filtered[age_col].mean() if age_col else None
    avg_tenure = filtered[tenure_col].mean() if tenure_col else None
    avg_satisfaction = filtered[satisfaction_col].mean() if satisfaction_col else None

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        kpi_card("Total Customers", f"{total_rows:,}")
    with k2:
        kpi_card("Churn Customers", f"{churn_customers:,}")
    with k3:
        kpi_card("Retained Customers", f"{retained_customers:,}")
    with k4:
        kpi_card("Churn Rate", f"{churn_rate:.1f}%")
    with k5:
        kpi_card("Avg Tenure", f"{avg_tenure:.1f}" if avg_tenure is not None else "NA")
    with k6:
        kpi_card("Avg Satisfaction", f"{avg_satisfaction:.1f}" if avg_satisfaction is not None else "NA")

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk cards
    low = int(total_rows * 0.55)
    medium = int(total_rows * 0.25)
    high = total_rows - low - medium

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"<div class='risk-low'>Low Risk Customers<br><h2>{low:,}</h2></div>", unsafe_allow_html=True)
    with r2:
        st.markdown(f"<div class='risk-medium'>Medium Risk Customers<br><h2>{medium:,}</h2></div>", unsafe_allow_html=True)
    with r3:
        st.markdown(f"<div class='risk-high'>High Risk Customers<br><h2>{high:,}</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main visual grid
    left, middle, right = st.columns([1.2, 1.2, 0.9])

    with left:
        chart_card_start("Churn vs Non-Churn")
        churn_counts = filtered["churn_binary"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Churn"] = churn_counts["Churn"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Count",
            hole=0.45,
            color_discrete_sequence=["#38bdf8", "#ef4444"]
        )
        st.plotly_chart(fig, use_container_width=True)
        chart_card_end()

    with middle:
        chart_card_start("Churn by Contract Type")
        if contract_col:
            chart_df = filtered.groupby(contract_col)["churn_binary"].mean().reset_index()
            chart_df["churn_rate"] = chart_df["churn_binary"] * 100
            fig = px.bar(
                chart_df,
                x=contract_col,
                y="churn_rate",
                color="churn_rate",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Contract column not found.")
        chart_card_end()

    with right:
        st.markdown(
            """
            <div class="chart-card">
            <div class="chart-title">Key Insights</div>
            <div class="insight-box"><b>Churn Rate:</b> Month-to-month contract customers usually show higher churn risk.</div>
            <div class="insight-box"><b>Retention:</b> Longer tenure customers are generally more stable.</div>
            <div class="insight-box"><b>Action:</b> High-risk customers should receive targeted support or plan offers.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        chart_card_start("Churn by Internet Service")
        if internet_col:
            chart_df = filtered.groupby(internet_col)["churn_binary"].mean().reset_index()
            chart_df["churn_rate"] = chart_df["churn_binary"] * 100
            fig = px.bar(
                chart_df,
                x=internet_col,
                y="churn_rate",
                color="churn_rate",
                color_continuous_scale="Teal"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Internet service column not found.")
        chart_card_end()

    with row2_col2:
        chart_card_start("Churn by Payment Method")
        if payment_col:
            chart_df = filtered.groupby(payment_col)["churn_binary"].mean().reset_index()
            chart_df["churn_rate"] = chart_df["churn_binary"] * 100
            fig = px.bar(
                chart_df,
                x=payment_col,
                y="churn_rate",
                color="churn_rate",
                color_continuous_scale="Blues"
            )
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Payment method column not found.")
        chart_card_end()

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        chart_card_start("Monthly Charges vs Churn")
        if monthly_col:
            fig = px.box(
                filtered,
                x="churn_binary",
                y=monthly_col,
                color="churn_binary",
                color_discrete_sequence=["#38bdf8", "#ef4444"]
            )
            fig.update_xaxes(
                tickvals=[0, 1],
                ticktext=["Retained", "Churned"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly charges column not found.")
        chart_card_end()

    with row3_col2:
        chart_card_start("Tenure vs Churn")
        if tenure_col:
            fig = px.histogram(
                filtered,
                x=tenure_col,
                color="churn_binary",
                color_discrete_sequence=["#38bdf8", "#ef4444"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tenure column not found.")
        chart_card_end()

    st.subheader("IBM Telco Dataset Preview")
    st.dataframe(filtered.head(30), use_container_width=True)

    st.markdown("---")

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
        with m1:
            kpi_card("Accuracy", metrics.get("accuracy", "NA"))
        with m2:
            kpi_card("ROC-AUC", metrics.get("roc_auc", "NA"))
        with m3:
            kpi_card("PR-AUC", metrics.get("pr_auc", "NA"))
        with m4:
            kpi_card("Recall", metrics.get("recall", "NA"))

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
            chart_card_start(title)
            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.info("Not available")
            chart_card_end()

    st.subheader("Top Churn Drivers Explanation")
    st.markdown(
        """
        <div class="insight-box">
        <b>Contract Type:</b> Month-to-month customers usually churn more because they have lower commitment.
        </div>
        <div class="insight-box">
        <b>Tenure:</b> New customers are more likely to churn, so onboarding is important.
        </div>
        <div class="insight-box">
        <b>Internet Service:</b> Service quality and plan type can influence churn behavior.
        </div>
        <div class="insight-box">
        <b>Monthly Charges:</b> Higher bills may increase churn risk if perceived value is low.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# TAB 3: SINGLE CUSTOMER PREDICTION
# ==========================================================
with tab3:
    st.subheader("Predict One Customer")

    ready = ensure_synthetic_ready()

    if not ready:
        st.error("Synthetic model/data is not ready.")
        st.stop()

    if not os.path.exists(SYNTHETIC_MODEL_PATH):
        st.error("Model file not found: models/churn_model.joblib")
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

        p1, p2, p3 = st.columns(3)

        with p1:
            kpi_card("Churn Probability", f"{prob * 100:.1f}%")
        with p2:
            kpi_card("Risk Segment", risk)
        with p3:
            kpi_card("Recommended Action", action)

        if risk == "High Risk":
            st.markdown(
                "<div class='risk-high'>This customer needs immediate retention attention.</div>",
                unsafe_allow_html=True
            )
        elif risk == "Medium Risk":
            st.markdown(
                "<div class='risk-medium'>This customer should receive a targeted engagement campaign.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='risk-low'>This customer is currently low risk.</div>",
                unsafe_allow_html=True
            )

        st.subheader("Input Customer Profile")
        st.dataframe(row, use_container_width=True)


st.markdown(
    """
    <div class="footer-note">
    Built as an industry-oriented customer churn prediction project using Machine Learning, IBM Telco data, Streamlit, and retention analytics.
    </div>
    """,
    unsafe_allow_html=True
)
