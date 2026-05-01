import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)


# ==========================================================
# PATH CONFIG
# ==========================================================
DATA_PATH = "data/telco_customer_churn.csv"
MODEL_PATH = "models/telco_churn_model.joblib"
OUTPUT_DIR = "outputs"
IMAGE_DIR = "images"


# ==========================================================
# PREMIUM CHART STYLE
# ==========================================================
BLUE = "#0F6E9C"
LIGHT_BLUE = "#8FD0F4"
DARK_BLUE = "#0B5A7C"
RED = "#EF5350"
GRID = "#D9E2EC"
TEXT = "#102A43"


def apply_premium_chart_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
    })


# ==========================================================
# DATA CLEANING
# ==========================================================
def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    return df


def find_target_column(df):
    possible_targets = [
        "churn",
        "customer_status",
        "churn_label",
        "churn_value",
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    raise ValueError(
        "Target column not found. Expected Churn, Customer Status, Churn Label, or Churn Value."
    )


def prepare_target(df, target_col):
    values = df[target_col].astype(str).str.lower().str.strip()

    if target_col == "customer_status":
        y = values.apply(lambda x: 1 if "churn" in x else 0)
    else:
        y = values.map({
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "churned": 1,
            "stayed": 0,
            "joined": 0
        })

    if y.isna().sum() > 0:
        unknown_values = values[y.isna()].unique()
        raise ValueError(
            f"Some target values could not be converted: {unknown_values}"
        )

    return y.astype(int)


def drop_unwanted_columns(df, target_col):
    leakage_cols = [
        target_col,
        "customer_id",
        "customerid",
        "churn_category",
        "churn_reason",
        "churn_score",
        "customer_status",
        "churn_label",
        "churn_value",
    ]

    location_or_id_cols = [
        "zip_code",
        "zip",
        "lat_long",
        "latitude",
        "longitude",
    ]

    # CLTV can be either useful or leakage depending on dataset.
    # For portfolio simplicity, we drop it to avoid target leakage concerns.
    business_post_outcome_cols = [
        "cltv"
    ]

    drop_cols = leakage_cols + location_or_id_cols + business_post_outcome_cols
    existing_cols = [col for col in drop_cols if col in df.columns]

    return df.drop(columns=existing_cols)


def convert_numeric_columns(X):
    """
    Safely converts columns that are mostly numeric.
    Text columns remain categorical.
    """
    X = X.copy()

    for col in X.columns:
        if X[col].dtype == "object":
            cleaned = (
                X[col]
                .astype(str)
                .str.strip()
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )

            converted = pd.to_numeric(cleaned, errors="coerce")

            # Convert only if most non-null values are numeric
            if converted.notna().mean() > 0.80:
                X[col] = converted
            else:
                X[col] = X[col].astype(str).str.strip()

    return X


# ==========================================================
# BUSINESS LOGIC
# ==========================================================
def assign_risk_segment(prob):
    if prob >= 0.70:
        return "Critical Risk"
    elif prob >= 0.50:
        return "High Risk"
    elif prob >= 0.25:
        return "Medium Risk"
    else:
        return "Low Risk"


def recommend_retention_action(row):
    prob = row["churn_probability"]

    if prob >= 0.70:
        return "Priority retention call + personalized discount"
    elif prob >= 0.50:
        return "Support follow-up + plan review"
    elif prob >= 0.25:
        return "Personalized engagement campaign"
    else:
        return "Regular engagement email"


def find_main_reason(row):
    if "contract" in row and str(row["contract"]).lower().strip() == "month-to-month":
        return "Month-to-month contract"
    elif "tenure_months" in row and row["tenure_months"] <= 3:
        return "Low tenure customer"
    elif "monthly_charges" in row and row["monthly_charges"] >= 80:
        return "High monthly charges"
    elif "tech_support" in row and str(row["tech_support"]).lower().strip() == "no":
        return "No tech support"
    elif "online_security" in row and str(row["online_security"]).lower().strip() == "no":
        return "No online security"
    elif "payment_method" in row and "electronic" in str(row["payment_method"]).lower():
        return "Electronic check payment risk"
    else:
        return "Multiple churn indicators"


def lift_at_k(y_true, y_proba, k=0.10):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)

    top_k_count = max(1, int(len(y_true) * k))
    top_indices = np.argsort(-y_proba)[:top_k_count]

    top_churn_rate = y_true.iloc[top_indices].mean()
    overall_churn_rate = y_true.mean()

    if overall_churn_rate == 0:
        return 0

    return top_churn_rate / overall_churn_rate


# ==========================================================
# PLOTS
# ==========================================================
def save_confusion_matrix(y_test, y_pred):
    apply_premium_chart_style()

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor=GRID,
        annot_kws={"size": 13, "weight": "bold", "color": TEXT}
    )

    plt.title("Confusion Matrix - IBM Telco Churn Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(
        f"{IMAGE_DIR}/telco_confusion_matrix.png",
        dpi=170,
        bbox_inches="tight"
    )
    plt.close()


def save_roc_curve(y_test, y_proba):
    apply_premium_chart_style()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_value = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(
        fpr,
        tpr,
        color=BLUE,
        linewidth=3,
        label=f"ROC-AUC = {roc_auc_value:.3f}"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="#94A3B8",
        linestyle="--",
        linewidth=2
    )

    plt.title("ROC Curve - IBM Telco Churn Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{IMAGE_DIR}/telco_roc_curve.png",
        dpi=170,
        bbox_inches="tight"
    )
    plt.close()


def save_pr_curve(y_test, y_proba):
    apply_premium_chart_style()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_value = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(
        recall,
        precision,
        color=BLUE,
        linewidth=3,
        label=f"PR-AUC = {pr_auc_value:.3f}"
    )

    plt.title("Precision-Recall Curve - IBM Telco Churn Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{IMAGE_DIR}/telco_pr_curve.png",
        dpi=170,
        bbox_inches="tight"
    )
    plt.close()


def save_feature_importance(model_pipeline):
    apply_premium_chart_style()

    try:
        feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = model_pipeline.named_steps["model"].feature_importances_

        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        feature_importance.to_csv(
            f"{OUTPUT_DIR}/telco_feature_importance.csv",
            index=False
        )

        top_features = feature_importance.head(15).sort_values(
            "importance",
            ascending=True
        )

        plt.figure(figsize=(9, 6))
        plt.barh(
            top_features["feature"],
            top_features["importance"],
            color=BLUE
        )

        plt.title("Top Feature Importance - IBM Telco Churn Model")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.grid(axis="x")
        plt.tight_layout()
        plt.savefig(
            f"{IMAGE_DIR}/telco_feature_importance.png",
            dpi=170,
            bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        print("Feature importance generation skipped:", str(e))


# ==========================================================
# MAIN TRAINING
# ==========================================================
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please upload telco_customer_churn.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    df = clean_column_names(df)

    target_col = find_target_column(df)
    y = prepare_target(df, target_col)

    X = drop_unwanted_columns(df, target_col)

    X = X.replace(" ", pd.NA)
    X = X.replace("", pd.NA)
    X = convert_numeric_columns(X)

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("Target column:", target_col)
    print("Rows:", len(df))
    print("Features used:", X.shape[1])
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print("Churn rate:", round(float(y.mean()), 4))

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=14,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "rows": int(len(df)),
        "features_used": int(X.shape[1]),
        "target_column": target_col,
        "churn_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "lift_at_10_percent": round(float(lift_at_k(y_test, y_proba, k=0.10)), 4)
    }

    print("\nIBM Telco Model Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    with open(f"{OUTPUT_DIR}/telco_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(model_pipeline, MODEL_PATH)

    # Save premium charts
    save_confusion_matrix(y_test, y_pred)
    save_roc_curve(y_test, y_proba)
    save_pr_curve(y_test, y_proba)
    save_feature_importance(model_pipeline)

    # Churn Watchlist
    scored = X_test.copy()
    scored["actual_churn"] = y_test.values
    scored["churn_probability"] = y_proba
    scored["risk_segment"] = scored["churn_probability"].apply(assign_risk_segment)
    scored["recommended_action"] = scored.apply(recommend_retention_action, axis=1)
    scored["main_reason"] = scored.apply(find_main_reason, axis=1)

    priority_columns = [
        "churn_probability",
        "risk_segment",
        "recommended_action",
        "main_reason",
        "actual_churn"
    ]

    remaining_columns = [col for col in scored.columns if col not in priority_columns]

    scored = scored[priority_columns + remaining_columns]
    scored = scored.sort_values("churn_probability", ascending=False)

    scored.head(50).to_csv(
        f"{OUTPUT_DIR}/telco_top_50_churn_watchlist.csv",
        index=False
    )

    print("\nSaved files:")
    print(f"- {MODEL_PATH}")
    print(f"- {OUTPUT_DIR}/telco_metrics.json")
    print(f"- {OUTPUT_DIR}/telco_top_50_churn_watchlist.csv")
    print(f"- {OUTPUT_DIR}/telco_feature_importance.csv")
    print(f"- {IMAGE_DIR}/telco_confusion_matrix.png")
    print(f"- {IMAGE_DIR}/telco_roc_curve.png")
    print(f"- {IMAGE_DIR}/telco_pr_curve.png")
    print(f"- {IMAGE_DIR}/telco_feature_importance.png")


if __name__ == "__main__":
    main()
