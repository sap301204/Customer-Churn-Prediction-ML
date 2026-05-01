import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


DATA_PATH = "data/telco_customer_churn.csv"
MODEL_PATH = "models/telco_churn_model.joblib"
OUTPUT_DIR = "outputs"
IMAGE_DIR = "images"


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
        "Target column not found. Please check if your dataset has Churn, Customer Status, Churn Label, or Churn Value."
    )


def prepare_target(df, target_col):
    df = df.copy()
    target_values = df[target_col].astype(str).str.lower().str.strip()

    if target_col == "customer_status":
        y = target_values.apply(lambda x: 1 if "churn" in x else 0)
    else:
        y = target_values.map({
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
        unknown_values = target_values[y.isna()].unique()
        raise ValueError(f"Some target values could not be converted: {unknown_values}")

    return y.astype(int)


def drop_unwanted_columns(df, target_col):
    leakage_cols = [
        target_col,
        "customer_id",
        "customerid",
        "churn_category",
        "churn_reason",
        "churn_score",
        "cltv",
        "customer_status",
        "churn_label",
        "churn_value",
    ]

    id_location_cols = [
        "zip_code",
        "zip",
        "lat_long",
        "latitude",
        "longitude",
    ]

    drop_cols = leakage_cols + id_location_cols
    existing_cols = [col for col in drop_cols if col in df.columns]

    return df.drop(columns=existing_cols)


def convert_numeric_columns(X):
    """
    Safely convert columns that are actually numeric.
    Keeps text columns as categorical.
    This avoids pandas errors and prevents destroying categorical features.
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

            # Convert only if most values are numeric
            if converted.notna().mean() > 0.80:
                X[col] = converted
            else:
                X[col] = X[col].astype(str).str.strip()

    return X


def add_retention_action(row):
    prob = row["churn_probability"]

    if prob >= 0.70:
        return "Priority retention call + personalized discount"
    elif prob >= 0.50:
        return "Support follow-up + plan review"
    elif prob >= 0.25:
        return "Personalized engagement campaign"
    else:
        return "Regular engagement email"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please upload telco_customer_churn.csv inside data folder."
        )

    df = pd.read_csv(DATA_PATH)
    df = clean_column_names(df)

    target_col = find_target_column(df)
    y = prepare_target(df, target_col)

    X = drop_unwanted_columns(df, target_col)

    # Replace blank values
    X = X.replace(" ", pd.NA)
    X = X.replace("", pd.NA)

    # Safe numeric conversion
    X = convert_numeric_columns(X)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("Target column:", target_col)
    print("Rows:", len(df))
    print("Features used:", X.shape[1])
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print("Churn rate:", round(y.mean(), 4))

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
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=12,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

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
    }

    print("\nModel Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    with open(f"{OUTPUT_DIR}/telco_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(pipeline, MODEL_PATH)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - IBM Telco Churn Model")
    plt.savefig(f"{IMAGE_DIR}/telco_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - IBM Telco Churn Model")
    plt.savefig(f"{IMAGE_DIR}/telco_roc_curve.png", bbox_inches="tight")
    plt.close()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title("Precision-Recall Curve - IBM Telco Churn Model")
    plt.savefig(f"{IMAGE_DIR}/telco_pr_curve.png", bbox_inches="tight")
    plt.close()

    # Feature Importance
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_

        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        feature_importance.to_csv(f"{OUTPUT_DIR}/telco_feature_importance.csv", index=False)

        top_features = feature_importance.head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features["feature"], top_features["importance"])
        plt.gca().invert_yaxis()
        plt.title("Top Feature Importance - IBM Telco Churn Model")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{IMAGE_DIR}/telco_feature_importance.png", bbox_inches="tight")
        plt.close()

    except Exception as e:
        print("Feature importance generation skipped:", str(e))

    # Churn Watchlist
    scored = X_test.copy()
    scored["actual_churn"] = y_test.values
    scored["churn_probability"] = y_proba

    scored["risk_segment"] = pd.cut(
        scored["churn_probability"],
        bins=[0, 0.25, 0.50, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True
    )

    scored["recommended_action"] = scored.apply(add_retention_action, axis=1)

    scored = scored.sort_values("churn_probability", ascending=False)
    scored.head(50).to_csv(f"{OUTPUT_DIR}/telco_top_50_churn_watchlist.csv", index=False)

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
