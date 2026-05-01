import json
import os
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent))
from features import add_synthetic_features, recommend_action, risk_segment
from pipeline import build_preprocessor

def lift_at_k(y_true, y_score, k=0.10):
    frame = pd.DataFrame({"y": y_true.values, "p": y_score})
    top_n = max(1, int(len(frame) * k))
    return frame.sort_values("p", ascending=False).head(top_n)["y"].mean() / frame["y"].mean()

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    if not os.path.exists("data/churn_frame.csv"):
        raise FileNotFoundError("Run: python src/generate_data.py --rows 5000")

    df = pd.read_csv("data/churn_frame.csv")
    df = add_synthetic_features(df)

    target = "churned_next_cycle"
    X = df.drop(columns=[target, "customer_id"])
    y = df[target]

    model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": "synthetic",
        "rows": len(df),
        "churn_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "lift_at_10_percent": round(float(lift_at_k(y_test, y_prob)), 4)
    }

    joblib.dump(model, "models/churn_model.joblib")
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    scored = X_test.copy()
    scored["actual_churn"] = y_test.values
    scored["churn_probability"] = y_prob
    scored["risk_segment"] = [risk_segment(p) for p in y_prob]
    scored["recommended_action"] = [recommend_action(p, row.to_dict()) for p, (_, row) in zip(y_prob, scored.iterrows())]
    scored.sort_values("churn_probability", ascending=False).head(50).to_csv("outputs/top_50_churn_watchlist.csv", index=False)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    fi.to_csv("outputs/feature_importance.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    plt.savefig("images/roc_curve.png", bbox_inches="tight", dpi=150)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall Curve")
    plt.savefig("images/pr_curve.png", bbox_inches="tight", dpi=150)
    plt.close()

    fi.head(15).sort_values("importance").plot.barh(x="feature", y="importance", legend=False)
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", bbox_inches="tight", dpi=150)
    plt.close()

    print("Synthetic model trained successfully")
    print(metrics)

if __name__ == "__main__":
    main()
