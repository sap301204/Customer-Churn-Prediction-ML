import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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

try:
    from features import add_features
    from pipeline import pre
except ImportError:
    from src.features import add_features
    from src.pipeline import pre


# ==========================================================
# PATH CONFIG
# ==========================================================
DATA_PATH = "data/churn_frame.csv"
MODEL_PATH = "models/churn_model.joblib"
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
    if "support_tickets" in row and row["support_tickets"] >= 3:
        return "High support issues"
    elif "last_payment_days_ago" in row and row["last_payment_days_ago"] >= 20:
        return "Payment delay"
    elif "monthly_usage_hours" in row and row["monthly_usage_hours"] <= 10:
        return "Low product usage"
    elif "tenure_months" in row and row["tenure_months"] <= 3:
        return "New customer churn risk"
    elif "billing_amount" in row and row["billing_amount"] >= 90:
        return "High billing amount"
    elif "nps_score" in row and row["nps_score"] <= 4:
        return "Low customer satisfaction"
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

    plt.title("Confusion Matrix - Churn Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/confusion_matrix.png", dpi=170, bbox_inches="tight")
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

    plt.title("ROC Curve - Churn Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/roc_curve.png", dpi=170, bbox_inches="tight")
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

    plt.title("Precision-Recall Curve - Churn Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/pr_curve.png", dpi=170, bbox_inches="tight")
    plt.close()


def save_feature_importance(model_pipeline):
    apply_premium_chart_style()

    try:
        feature_names = model_pipeline.named_steps["pre"].get_feature_names_out()
        importances = model_pipeline.named_steps["model"].feature_importances_

        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        feature_importance.to_csv(
            f"{OUTPUT_DIR}/feature_importance.csv",
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

        plt.title("Top Feature Importance - Churn Model")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.grid(axis="x")
        plt.tight_layout()
        plt.savefig(
            f"{IMAGE_DIR}/feature_importance.png",
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
            f"Dataset not found at {DATA_PATH}. Run generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    target = "churned_next_cycle"

    drop_columns = [
        target,
        "customer_id",
        "cycle_start",
        "cycle_end"
    ]

    existing_drop_columns = [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=existing_drop_columns)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model_pipeline = Pipeline(steps=[
        ("pre", pre),
        ("model", model)
    ])

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "rows": int(len(df)),
        "features_used": int(X.shape[1]),
        "churn_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "lift_at_10_percent": round(float(lift_at_k(y_test, y_proba, k=0.10)), 4)
    }

    print("\nSynthetic Churn Model Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(model_pipeline, MODEL_PATH)

    # Save charts
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
        f"{OUTPUT_DIR}/top_50_churn_watchlist.csv",
        index=False
    )

    print("\nSaved files:")
    print(f"- {MODEL_PATH}")
    print(f"- {OUTPUT_DIR}/metrics.json")
    print(f"- {OUTPUT_DIR}/top_50_churn_watchlist.csv")
    print(f"- {OUTPUT_DIR}/feature_importance.csv")
    print(f"- {IMAGE_DIR}/confusion_matrix.png")
    print(f"- {IMAGE_DIR}/roc_curve.png")
    print(f"- {IMAGE_DIR}/pr_curve.png")
    print(f"- {IMAGE_DIR}/feature_importance.png")


if __name__ == "__main__":
    main()
