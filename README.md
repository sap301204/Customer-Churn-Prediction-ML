# Customer Churn Prediction Model

Industry-oriented ML project for predicting customer churn, creating risk segments, and recommending retention actions.

## Final Structure

```text
Customer-Churn-Prediction/
├── data/
│   ├── churn_frame.csv
│   ├── telco_customer_churn.csv
│   └── README.md
├── notebooks/
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── pipeline.py
│   ├── train_model.py
│   └── train_telco_model.py
├── models/
│   ├── churn_model.joblib
│   └── telco_churn_model.joblib
├── outputs/
│   ├── metrics.json
│   ├── top_50_churn_watchlist.csv
│   ├── feature_importance.csv
│   ├── telco_metrics.json
│   ├── telco_top_50_churn_watchlist.csv
│   └── telco_feature_importance.csv
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── feature_importance.png
│   ├── telco_confusion_matrix.png
│   ├── telco_roc_curve.png
│   ├── telco_pr_curve.png
│   └── telco_feature_importance.png
├── serving/
│   └── app.py
├── apps/web/
│   └── Next.js dashboard demo
├── app.py
├── streamlit_app.py
├── Dockerfile
├── README.md
├── requirements.txt
└── main.py
```

## Dataset Support

### Synthetic Dataset

```bash
python src/generate_data.py --rows 5000
python src/train_model.py
```

### IBM Telco Dataset

Place your IBM Telco CSV here:

```text
data/telco_customer_churn.csv
```

Then run:

```bash
python src/train_telco_model.py
```

## Streamlit Dashboard

For Streamlit Cloud, select this main file path:

```text
streamlit_app.py
```

or

```text
app.py
```

Run locally:

```bash
streamlit run streamlit_app.py
```

## FastAPI

```bash
uvicorn serving.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Business Value

This project helps companies:
- identify high-risk customers,
- reduce churn,
- prioritize retention campaigns,
- recommend business actions,
- improve customer lifetime value.

## Interview Summary

I built a customer churn prediction system using Python, pandas, scikit-learn, Streamlit, and FastAPI. The project supports both synthetic churn simulation and the IBM Telco public churn dataset. It produces churn probability, risk segments, a top-50 churn watchlist, evaluation charts, and retention action recommendations.
