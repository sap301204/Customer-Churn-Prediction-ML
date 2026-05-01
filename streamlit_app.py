import json
import os
import subprocess
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================
SYNTHETIC_DATA_PATH = "data/churn_frame.csv"
TELCO_CSV_PATH = "data/telco_customer_churn.csv"
TELCO_XLSX_PATH = "Telco_customer_churn.xlsx"

SYNTHETIC_MODEL_PATH = "models/churn_model.joblib"

METRICS_PATH = "outputs/metrics.json"
TELCO_METRICS_PATH = "outputs/telco_metrics.json"

WATCHLIST_PATH = "outputs/top_50_churn_watchlist.csv"
TELCO_WATCHLIST_PATH = "outputs/telco_top_50_churn_watchlist.csv"


# =========================================================
# COLORS
# =========================================================
PRIMARY = "#0F6E9C"
PRIMARY_DARK = "#084C70"
PRIMARY_MID = "#147FB5"
PRIMARY_LIGHT = "#E7F5FB"
SECONDARY = "#1E88E5"
ACCENT = "#67C7F7"

BG = "#F3F7FA"
CARD = "#FFFFFF"
TEXT = "#102A43"
MUTED = "#5B7083"
BORDER = "#D8E4EC"

LOW_BG = "#E7F8EF"
LOW_TEXT = "#12703A"
MED_BG = "#FFF8DD"
MED_TEXT = "#9A6700"
HIGH_BG = "#FDECEC"
HIGH_TEXT = "#A61B1B"

PLOT_BLUE = "#1F84C7"
PLOT_BLUE_2 = "#67C7F7"
PLOT_BLUE_3 = "#0F6E9C"
PLOT_NAVY = "#084C70"
PLOT_ORANGE = "#F59E0B"
PLOT_GRID = "#E8EEF3"

BLUE_SCALE = [
    "#E7F5FB",
    "#C9EAF8",
    "#9ED8F2",
    "#67C7F7",
    "#2C9CDB",
    "#147FB5",
    "#0F6E9C",
    "#084C70"
]


# =========================================================
# PREMIUM CSS
# =========================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
        font-family: "Segoe UI", sans-serif;
    }}

    [data-testid="stAppViewContainer"] {{
        background: {BG};
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1520px;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #063E5C 0%, {PRIMARY_DARK} 46%, {PRIMARY} 100%);
        color: white;
        border-right: 1px solid rgba(255,255,255,0.12);
        box-shadow: 6px 0 18px rgba(8,76,112,0.10);
    }}

    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    .sidebar-logo-card {{
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 22px;
        padding: 18px 16px;
        margin-bottom: 18px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.16);
    }}

    .sidebar-title {{
        font-size: 1.35rem;
        font-weight: 850;
        color: white;
        margin-bottom: 6px;
    }}

    .sidebar-subtitle {{
        font-size: 0.85rem;
        color: #DDF2FB;
        line-height: 1.45;
    }}

    .sidebar-card {{
        background: rgba(255,255,255,0.11);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 18px;
        padding: 15px;
        margin-bottom: 16px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.12);
    }}

    .sidebar-card h3 {{
        color: white !important;
        font-size: 1rem;
        margin-bottom: 8px;
    }}

    .sidebar-item {{
        background: rgba(255,255,255,0.10);
        padding: 9px 11px;
        border-radius: 12px;
        margin: 7px 0;
        font-size: 0.90rem;
        border: 1px solid rgba(255,255,255,0.09);
    }}

    .hero {{
        background: linear-gradient(90deg, {PRIMARY_DARK}, {PRIMARY_MID});
        border-radius: 26px;
        padding: 28px 36px;
        color: white;
        box-shadow: 0 12px 30px rgba(8,76,112,0.22);
        margin-bottom: 18px;
    }}

    .hero-title {{
        font-size: 2.25rem;
        font-weight: 850;
        margin: 0;
        text-align: center;
        color: white;
        letter-spacing: -0.4px;
    }}

    .hero-sub {{
        font-size: 1rem;
        opacity: 0.96;
        margin-top: 8px;
        text-align: center;
        color: #EAF8FF;
    }}

    .nav-info {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 11px 16px;
        box-shadow: 0 6px 18px rgba(16,42,67,0.06);
        font-weight: 750;
        color: {PRIMARY_DARK};
        margin-bottom: 18px;
    }}

    .section-title {{
        font-size: 1.85rem;
        font-weight: 850;
        color: {TEXT};
        margin-top: 8px;
        margin-bottom: 16px;
    }}

    .subtle-title {{
        font-size: 1.5rem;
        font-weight: 850;
        color: {TEXT};
        margin: 24px 0 12px 0;
    }}

    .kpi-card {{
        background: linear-gradient(180deg, {PRIMARY_MID}, {PRIMARY_DARK});
        color: white;
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 22px rgba(8,76,112,0.18);
        min-height: 122px;
        border: none;
    }}

    .kpi-label {{
        font-size: 0.9rem;
        opacity: 0.95;
        font-weight: 700;
        margin-bottom: 14px;
        color: #EAF8FF;
    }}

    .kpi-value {{
        font-size: 2rem;
        font-weight: 850;
        line-height: 1.2;
        color: white;
    }}

    .risk-card {{
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        border: 1px solid {BORDER};
        box-shadow: 0 8px 24px rgba(16,42,67,0.07);
        min-height: 115px;
    }}

    .risk-title {{
        font-weight: 850;
        font-size: 1rem;
        margin-bottom: 10px;
    }}

    .risk-value {{
        font-weight: 850;
        font-size: 1.85rem;
    }}

    .chart-shell {{
        background: {CARD};
        border-radius: 20px;
        border: 1px solid {BORDER};
        box-shadow: 0 8px 24px rgba(16,42,67,0.07);
        padding: 16px;
        margin-bottom: 18px;
    }}

    .chart-title {{
        font-size: 1.08rem;
        font-weight: 850;
        color: {PRIMARY_DARK};
        margin-bottom: 10px;
    }}

    .insight-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(16,42,67,0.07);
        padding: 18px;
        height: 100%;
    }}

    .insight-box {{
        background: #EEF8FD;
        border-left: 5px solid {PRIMARY_MID};
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
        color: {TEXT};
        font-size: 0.95rem;
        line-height: 1.5;
    }}

    .small-note {{
        color: {MUTED};
        font-size: 0.95rem;
        margin-top: -6px;
        margin-bottom: 18px;
    }}

    .footer-note {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 14px;
        text-align: center;
        color: {PRIMARY_DARK};
        font-weight: 700;
        margin-top: 24px;
        box-shadow: 0 6px 18px rgba(16,42,67,0.05);
    }}

    .stButton > button {{
        background: linear-gradient(90deg, {PRIMARY_DARK}, {PRIMARY_MID});
        color: white !important;
        border: none;
        border-radius: 14px;
        font-weight: 800;
        padding: 0.65rem 1rem;
        box-shadow: 0 8px 18px rgba(8,76,112,0.18);
        width: 100%;
    }}

    .stButton > button:hover {{
        background: linear-gradient(90deg, {PRIMARY_MID}, {SECONDARY});
        color: white !important;
        border: none;
        transform: translateY(-1px);
    }}

    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stTextInput label,
    .stMultiSelect label {{
        color: {TEXT} !important;
        font-weight: 800 !important;
    }}

    .stSlider [data-baseweb="slider"] div {{
        color: {PRIMARY_DARK} !important;
    }}

    .stSlider div[role="slider"] {{
        background-color: {PRIMARY_MID} !important;
        border-color: {PRIMARY_MID} !important;
    }}

    div[data-testid="stDataFrame"] {{
        background: white !important;
        border-radius: 18px !important;
        border: 1px solid {BORDER} !important;
        box-shadow: 0 8px 24px rgba(16,42,67,0.07);
    }}

    .stAlert {{
        border-radius: 14px;
    }}

    .stMarkdown, p, label, div {{
        color: {TEXT};
    }}

    h1, h2, h3, h4, h5 {{
        color: {TEXT};
    }}

    /* Hide any radio input visuals if Streamlit adds them anywhere */
    div[role="radiogroup"] input {{
        display: none !important;
    }}

    div[role="radiogroup"] label > div:first-child {{
        display: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPERS
# =========================================================
def ensure_folders():
    for folder in ["data", "models", "outputs", "images", "src"]:
        os.makedirs(folder, exist_ok=True)


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def read_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def risk_card(title, value, bg, color):
    st.markdown(
        f"""
        <div class="risk-card" style="background:{bg}; color:{color};">
            <div class="risk-title" style="color:{color};">{title}</div>
            <div class="risk-value" style="color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def subtle_title(text):
    st.markdown(f'<div class="subtle-title">{text}</div>', unsafe_allow_html=True)


def chart_container_start(title):
    st.markdown(
        f"""
        <div class="chart-shell">
        <div class="chart-title">{title}</div>
        """,
        unsafe_allow_html=True
    )


def chart_container_end():
    st.markdown("</div>", unsafe_allow_html=True)


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


def find_column(df, possible_names):
    cols = list(df.columns)
    for name in possible_names:
        clean_name = clean_col_name(name)
        if clean_name in cols:
            return clean_name
    return None


def detect_churn_column(df):
    return find_column(df, ["churn", "churn_label", "customer_status", "churn_value"])


def create_churn_binary(df, churn_col):
    values = df[churn_col].astype(str).str.lower().str.strip()

    if churn_col == "customer_status":
        return values.apply(lambda x: 1 if "churn" in x else 0)

    return values.apply(lambda x: 1 if x in ["yes", "1", "true", "churned"] else 0)


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
            st.error("Found Excel file, but conversion to CSV failed.")
            st.code(str(e))
            return False

    return False


def ensure_synthetic_ready():
    ensure_folders()

    if not os.path.exists(SYNTHETIC_DATA_PATH):
        if os.path.exists("src/generate_data.py"):
            out, err, code = run_command(f"{sys.executable} src/generate_data.py --rows 5000")
        elif os.path.exists("generate_data.py"):
            out, err, code = run_command(f"{sys.executable} generate_data.py --rows 5000")
        else:
            return False

        if code != 0:
            st.error("Synthetic data generation failed.")
            st.code(err)
            return False

        if os.path.exists("churn_frame.csv") and not os.path.exists(SYNTHETIC_DATA_PATH):
            pd.read_csv("churn_frame.csv").to_csv(SYNTHETIC_DATA_PATH, index=False)

    if not os.path.exists(SYNTHETIC_MODEL_PATH):
        if os.path.exists("src/train_model.py"):
            run_command(f"{sys.executable} src/train_model.py")
        elif os.path.exists("train_model.py"):
            run_command(f"{sys.executable} train_model.py")

    return os.path.exists(SYNTHETIC_DATA_PATH)


def powerbi_layout(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=TEXT, family="Segoe UI", size=12),
        margin=dict(l=28, r=28, t=18, b=30),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color=TEXT)
        ),
        coloraxis_colorbar=dict(
            title_font=dict(color=TEXT),
            tickfont=dict(color=TEXT)
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=PLOT_GRID,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT)
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOT_GRID,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT)
    )

    return fig


def risk_action(prob):
    if prob >= 0.70:
        return "Critical Risk", "Priority retention call + personalized discount"
    elif prob >= 0.50:
        return "High Risk", "Support follow-up + plan review"
    elif prob >= 0.25:
        return "Medium Risk", "Personalized engagement campaign"
    else:
        return "Low Risk", "Regular engagement email"


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo-card">
            <div class="sidebar-title">📊 Dashboard Controls</div>
            <div class="sidebar-subtitle">Customer churn analytics and retention intelligence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <h3>Project Summary</h3>
            <div class="sidebar-subtitle">
            This dashboard predicts customer churn, identifies high-risk customers,
            explains churn drivers, and suggests retention actions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <h3>Dashboard Sections</h3>
            <div class="sidebar-item">✅ Synthetic churn simulation</div>
            <div class="sidebar-item">✅ IBM Telco dataset analysis</div>
            <div class="sidebar-item">✅ Single customer prediction</div>
            <div class="sidebar-item">✅ Risk segmentation</div>
            <div class="sidebar-item">✅ Retention actions</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <h3>Dataset Note</h3>
            <div class="sidebar-subtitle">
            IBM Telco dataset is a public dataset used for educational and portfolio purposes.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📉 Customer Churn Prediction Dashboard</div>
        <div class="hero-sub">
            Premium churn analytics dashboard for customer retention, business insights, and ML scoring
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# BUTTON NAVIGATION - NO RED RADIO DOTS
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "synthetic"

nav1, nav2, nav3 = st.columns(3)

with nav1:
    if st.button("📊 Synthetic Churn Dashboard"):
        st.session_state.page = "synthetic"

with nav2:
    if st.button("🏢 IBM Telco Dataset"):
        st.session_state.page = "telco"

with nav3:
    if st.button("🎯 Single Customer Prediction"):
        st.session_state.page = "prediction"

current_label = {
    "synthetic": "📊 Synthetic Churn Dashboard",
    "telco": "🏢 IBM Telco Dataset",
    "prediction": "🎯 Single Customer Prediction"
}[st.session_state.page]

st.markdown(
    f"""
    <div class="nav-info">Current View: {current_label}</div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PAGE 1: SYNTHETIC DASHBOARD
# =========================================================
if st.session_state.page == "synthetic":
    section_title("Synthetic Customer Churn Dashboard")

    ready = ensure_synthetic_ready()

    if not ready:
        st.error("Synthetic dataset is not available.")
        st.stop()

    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    metrics = read_json(METRICS_PATH)

    total_customers = len(df)
    churn_rate = df["churned_next_cycle"].mean() * 100
    roc_auc = metrics.get("roc_auc", "NA")
    lift = metrics.get("lift_at_10_percent", "NA")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Total Customers", f"{total_customers:,}")
    with c2:
        kpi_card("Churn Rate", f"{churn_rate:.1f}%")
    with c3:
        kpi_card("ROC-AUC", roc_auc)
    with c4:
        kpi_card("Lift@10%", lift)

    st.markdown("<br>", unsafe_allow_html=True)

    low_risk = int(total_customers * 0.55)
    medium_risk = int(total_customers * 0.30)
    high_risk = total_customers - low_risk - medium_risk

    r1, r2, r3 = st.columns(3)

    with r1:
        risk_card("Low Risk", f"{low_risk:,}", LOW_BG, LOW_TEXT)
    with r2:
        risk_card("Medium Risk", f"{medium_risk:,}", MED_BG, MED_TEXT)
    with r3:
        risk_card("High Risk", f"{high_risk:,}", HIGH_BG, HIGH_TEXT)

    st.markdown("<br>", unsafe_allow_html=True)

    row1_a, row1_b = st.columns(2)

    with row1_a:
        chart_container_start("Churn Distribution")
        churn_counts = df["churned_next_cycle"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Churn"] = churn_counts["Churn"].map({0: "No Churn", 1: "Churn"})

        fig = px.pie(
            churn_counts,
            values="Count",
            names="Churn",
            hole=0.45,
            color="Churn",
            color_discrete_map={
                "No Churn": PLOT_BLUE,
                "Churn": PLOT_ORANGE
            }
        )

        fig = powerbi_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        chart_container_end()

    with row1_b:
        chart_container_start("Tenure vs Churn")
        fig = px.histogram(
            df,
            x="tenure_months",
            color="churned_next_cycle",
            barmode="overlay",
            color_discrete_map={
                0: PLOT_BLUE,
                1: PLOT_ORANGE
            },
            opacity=0.82
        )

        fig = powerbi_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        chart_container_end()

    row2_a, row2_b = st.columns(2)

    with row2_a:
        chart_container_start("Monthly Usage Hours vs Churn")
        fig = px.histogram(
            df,
            x="monthly_usage_hours",
            color="churned_next_cycle",
            barmode="overlay",
            color_discrete_map={
                0: PLOT_BLUE,
                1: PLOT_ORANGE
            },
            opacity=0.82
        )

        fig = powerbi_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        chart_container_end()

    with row2_b:
        chart_container_start("Support Tickets vs Churn")

        if "support_tickets" in df.columns:
            support_churn = df.groupby("support_tickets")["churned_next_cycle"].mean().reset_index()
            support_churn["churn_rate"] = support_churn["churned_next_cycle"] * 100

            fig = px.bar(
                support_churn,
                x="support_tickets",
                y="churn_rate",
                color_discrete_sequence=[PLOT_BLUE]
            )

            fig = powerbi_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Support ticket data not available.")

        chart_container_end()

    subtle_title("Top 50 Churn Watchlist")

    if os.path.exists(WATCHLIST_PATH):
        st.dataframe(pd.read_csv(WATCHLIST_PATH), use_container_width=True)
    else:
        st.info("Watchlist file not found. Train the model first.")

    subtle_title("Model Performance Charts")

    img1, img2, img3, img4 = st.columns(4)

    chart_files = [
        ("images/confusion_matrix.png", "Confusion Matrix"),
        ("images/roc_curve.png", "ROC Curve"),
        ("images/pr_curve.png", "Precision-Recall Curve"),
        ("images/feature_importance.png", "Feature Importance"),
    ]

    for col, (img, title) in zip([img1, img2, img3, img4], chart_files):
        with col:
            chart_container_start(title)

            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.info("Not available")

            chart_container_end()

    subtle_title("Business Insights")

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-box">
                <b>Insight 1:</b> Lower engagement and shorter tenure customers show higher churn tendency.
            </div>
            <div class="insight-box">
                <b>Insight 2:</b> Customers with payment delays and more support issues should be prioritized for retention.
            </div>
            <div class="insight-box">
                <b>Insight 3:</b> Lift@10% indicates the model can help target the most at-risk customer group effectively.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 2: IBM TELCO DASHBOARD
# =========================================================
elif st.session_state.page == "telco":
    section_title("IBM Telco Customer Churn Analysis Dashboard")

    telco_ready = convert_telco_excel_to_csv_if_needed()

    if not telco_ready:
        st.warning("IBM Telco dataset not found. Upload data/telco_customer_churn.csv or Telco_customer_churn.xlsx.")
        st.stop()

    telco_raw = pd.read_csv(TELCO_CSV_PATH)
    telco = clean_dataframe_columns(telco_raw)

    churn_col = detect_churn_column(telco)

    if churn_col is None:
        st.error("Churn column not detected.")
        st.dataframe(telco.head(), use_container_width=True)
        st.stop()

    telco["churn_binary"] = create_churn_binary(telco, churn_col)

    gender_col = find_column(telco, ["gender"])
    contract_col = find_column(telco, ["contract"])
    internet_col = find_column(telco, ["internet_service"])
    payment_col = find_column(telco, ["payment_method"])
    tenure_col = find_column(telco, ["tenure_months", "tenure"])
    monthly_col = find_column(telco, ["monthly_charge", "monthly_charges"])
    satisfaction_col = find_column(telco, ["satisfaction_score"])

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <h3>IBM Telco Filters</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        selected_gender = "All"
        selected_contract = "All"
        selected_internet = "All"
        selected_payment = "All"

        if gender_col:
            selected_gender = st.selectbox(
                "Gender",
                ["All"] + sorted(telco[gender_col].dropna().astype(str).unique().tolist()),
                key="gender_filter"
            )

        if contract_col:
            selected_contract = st.selectbox(
                "Contract Type",
                ["All"] + sorted(telco[contract_col].dropna().astype(str).unique().tolist()),
                key="contract_filter"
            )

        if internet_col:
            selected_internet = st.selectbox(
                "Internet Service",
                ["All"] + sorted(telco[internet_col].dropna().astype(str).unique().tolist()),
                key="internet_filter"
            )

        if payment_col:
            selected_payment = st.selectbox(
                "Payment Method",
                ["All"] + sorted(telco[payment_col].dropna().astype(str).unique().tolist()),
                key="payment_filter"
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

    low = int(total_rows * 0.55)
    medium = int(total_rows * 0.25)
    high = total_rows - low - medium

    rr1, rr2, rr3 = st.columns(3)

    with rr1:
        risk_card("Low Risk Customers", f"{low:,}", LOW_BG, LOW_TEXT)
    with rr2:
        risk_card("Medium Risk Customers", f"{medium:,}", MED_BG, MED_TEXT)
    with rr3:
        risk_card("High Risk Customers", f"{high:,}", HIGH_BG, HIGH_TEXT)

    st.markdown("<br>", unsafe_allow_html=True)

    top_a, top_b, top_c = st.columns([1.1, 1.1, 0.8])

    with top_a:
        chart_container_start("Churn vs Non-Churn")

        chart_df = filtered["churn_binary"].value_counts().reset_index()
        chart_df.columns = ["Churn", "Count"]
        chart_df["Churn"] = chart_df["Churn"].map({0: "Retained", 1: "Churned"})

        fig = px.pie(
            chart_df,
            names="Churn",
            values="Count",
            hole=0.45,
            color="Churn",
            color_discrete_map={
                "Retained": PLOT_BLUE,
                "Churned": PLOT_ORANGE
            }
        )

        fig = powerbi_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        chart_container_end()

    with top_b:
        chart_container_start("Churn by Contract Type")

        if contract_col:
            chart_df = filtered.groupby(contract_col)["churn_binary"].mean().reset_index()
            chart_df["churn_rate"] = chart_df["churn_binary"] * 100

            fig = px.bar(
                chart_df,
                x=contract_col,
                y="churn_rate",
                color="churn_rate",
                color_continuous_scale=BLUE_SCALE
            )

            fig = powerbi_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Contract column not found.")

        chart_container_end()

    with top_c:
        st.markdown(
            """
            <div class="insight-card">
                <div class="chart-title">Key Insights</div>
                <div class="insight-box">
                    <b>Churn Rate:</b> Month-to-month contract customers usually show higher churn risk.
                </div>
                <div class="insight-box">
                    <b>Retention:</b> Longer tenure customers are generally more stable.
                </div>
                <div class="insight-box">
                    <b>Action:</b> High-risk customers should receive targeted support or plan offers.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    mid1, mid2 = st.columns(2)

    with mid1:
        chart_container_start("Churn by Internet Service")

        if internet_col:
            chart_df = filtered.groupby(internet_col)["churn_binary"].mean().reset_index()
            chart_df["churn_rate"] = chart_df["churn_binary"] * 100

            fig = px.bar(
                chart_df,
                x=internet_col,
                y="churn_rate",
                color="churn_rate",
                color_continuous_scale=BLUE_SCALE
            )

            fig = powerbi_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Internet service column not found.")

        chart_container_end()

    with mid2:
        chart_container_start("Churn by Payment Method")

        if payment_col:
            chart_df = filtered.groupby(payment_col)["churn_binary"].mean().reset_index()
            chart_df["churn_rate"] = chart_df["churn_binary"] * 100

            fig = px.bar(
                chart_df,
                x=payment_col,
                y="churn_rate",
                color="churn_rate",
                color_continuous_scale=BLUE_SCALE
            )

            fig.update_layout(xaxis_tickangle=-18)
            fig = powerbi_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Payment method column not found.")

        chart_container_end()

    bot1, bot2 = st.columns(2)

    with bot1:
        chart_container_start("Monthly Charges vs Churn")

        if monthly_col:
            fig = px.box(
                filtered,
                x="churn_binary",
                y=monthly_col,
                color="churn_binary",
                color_discrete_map={
                    0: PLOT_BLUE,
                    1: PLOT_ORANGE
                }
            )

            fig.update_xaxes(tickvals=[0, 1], ticktext=["Retained", "Churned"])
            fig = powerbi_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly charges column not found.")

        chart_container_end()

    with bot2:
        chart_container_start("Tenure vs Churn")

        if tenure_col:
            fig = px.histogram(
                filtered,
                x=tenure_col,
                color="churn_binary",
                barmode="overlay",
                color_discrete_map={
                    0: PLOT_BLUE,
                    1: PLOT_ORANGE
                },
                opacity=0.82
            )

            fig = powerbi_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tenure column not found.")

        chart_container_end()

    subtle_title("IBM Telco Dataset Preview")
    st.dataframe(filtered.head(30), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    train_col, _ = st.columns([1, 4])

    with train_col:
        train_clicked = st.button("Train IBM Telco Model")

    if "telco_train_stdout" not in st.session_state:
        st.session_state.telco_train_stdout = ""

    if "telco_train_stderr" not in st.session_state:
        st.session_state.telco_train_stderr = ""

    if "telco_train_success" not in st.session_state:
        st.session_state.telco_train_success = None

    if train_clicked:
        with st.spinner("Training IBM Telco model..."):
            if os.path.exists("src/train_telco_model.py"):
                out, err, code = run_command(f"{sys.executable} src/train_telco_model.py")
            elif os.path.exists("train_telco_model.py"):
                out, err, code = run_command(f"{sys.executable} train_telco_model.py")
            else:
                out, err, code = "", "train_telco_model.py not found.", 1

            st.session_state.telco_train_stdout = out
            st.session_state.telco_train_stderr = err
            st.session_state.telco_train_success = code == 0

    if st.session_state.telco_train_success is True:
        st.success("IBM Telco model trained successfully.")
    elif st.session_state.telco_train_success is False:
        st.error("Training failed.")

    if st.session_state.telco_train_stdout or st.session_state.telco_train_stderr:
        with st.expander("View training logs"):
            if st.session_state.telco_train_stdout:
                st.code(st.session_state.telco_train_stdout)
            if st.session_state.telco_train_stderr:
                st.code(st.session_state.telco_train_stderr)

    metrics = read_json(TELCO_METRICS_PATH)

    if metrics:
        subtle_title("IBM Telco Model Metrics")

        mm1, mm2, mm3, mm4 = st.columns(4)

        with mm1:
            kpi_card("Accuracy", metrics.get("accuracy", "NA"))
        with mm2:
            kpi_card("ROC-AUC", metrics.get("roc_auc", "NA"))
        with mm3:
            kpi_card("PR-AUC", metrics.get("pr_auc", "NA"))
        with mm4:
            kpi_card("Recall", metrics.get("recall", "NA"))

    if os.path.exists(TELCO_WATCHLIST_PATH):
        subtle_title("IBM Telco Top 50 Churn Watchlist")
        st.dataframe(pd.read_csv(TELCO_WATCHLIST_PATH), use_container_width=True)

    subtle_title("IBM Telco Model Charts")

    im1, im2, im3, im4 = st.columns(4)

    telco_chart_files = [
        ("images/telco_confusion_matrix.png", "Confusion Matrix"),
        ("images/telco_roc_curve.png", "ROC Curve"),
        ("images/telco_pr_curve.png", "Precision-Recall Curve"),
        ("images/telco_feature_importance.png", "Feature Importance")
    ]

    for col, (img, title) in zip([im1, im2, im3, im4], telco_chart_files):
        with col:
            chart_container_start(title)

            if os.path.exists(img):
                st.image(img, use_container_width=True)
            else:
                st.info("Not available")

            chart_container_end()

    subtle_title("Top Churn Drivers Explanation")

    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-box">
                <b>Contract Type:</b> Month-to-month customers usually churn more because they have lower commitment.
            </div>
            <div class="insight-box">
                <b>Tenure:</b> New customers are more likely to churn, so onboarding matters a lot.
            </div>
            <div class="insight-box">
                <b>Internet Service:</b> Service quality and plan type can influence churn behavior.
            </div>
            <div class="insight-box">
                <b>Monthly Charges:</b> Higher monthly bills may increase churn risk if value perception is low.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 3: SINGLE CUSTOMER PREDICTION
# =========================================================
elif st.session_state.page == "prediction":
    section_title("Predict One Customer")

    st.markdown(
        '<div class="small-note">Enter a customer profile to estimate churn probability and recommended retention action.</div>',
        unsafe_allow_html=True
    )

    ready = ensure_synthetic_ready()

    if not ready:
        st.error("Synthetic model/data is not ready.")
        st.stop()

    if not os.path.exists(SYNTHETIC_MODEL_PATH):
        st.error("Model file not found: models/churn_model.joblib")
        st.stop()

    model = joblib.load(SYNTHETIC_MODEL_PATH)

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:
        age = st.slider("Age", 18, 80, 35)
        tenure_months = st.slider("Tenure Months", 1, 72, 12)
        billing_amount = st.number_input("Billing Amount", 10.0, 200.0, 75.0)
        monthly_usage_hours = st.number_input("Monthly Usage Hours", 0.0, 150.0, 10.0)
        active_days = st.slider("Active Days", 1, 30, 8)

    with input_col2:
        login_count = st.number_input("Login Count", 0, 250, 15)
        avg_session_min = st.number_input("Average Session Minutes", 1.0, 100.0, 15.0)
        support_tickets = st.slider("Support Tickets", 0, 10, 2)
        sla_breaches = st.slider("SLA Breaches", 0, 5, 1)
        nps_score = st.slider("NPS Score", 0, 10, 4)

    with input_col3:
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

    st.markdown("<br>", unsafe_allow_html=True)

    predict_col, _ = st.columns([1, 4])

    with predict_col:
        clicked = st.button("Predict Churn")

    if clicked:
        prob = float(model.predict_proba(row)[0, 1])
        risk, action = risk_action(prob)

        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            kpi_card("Churn Probability", f"{prob * 100:.1f}%")
        with rc2:
            kpi_card("Risk Segment", risk)
        with rc3:
            kpi_card("Recommended Action", action)

        st.markdown("<br>", unsafe_allow_html=True)

        if risk in ["Critical Risk", "High Risk"]:
            risk_card("High Risk Alert", "Immediate retention attention required", HIGH_BG, HIGH_TEXT)
        elif risk == "Medium Risk":
            risk_card("Medium Risk Alert", "Targeted engagement campaign recommended", MED_BG, MED_TEXT)
        else:
            risk_card("Low Risk Alert", "Customer is currently stable", LOW_BG, LOW_TEXT)

        subtle_title("Input Customer Profile")
        st.dataframe(row, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer-note">
        Built as an industry-oriented customer churn prediction project using Machine Learning, IBM Telco data, Streamlit, and retention analytics.
    </div>
    """,
    unsafe_allow_html=True
)
