# Credit Risk Prediction with Automated Drift Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

A production-grade machine learning system that predicts loan default risk and monitors data drift to ensure model reliability over time. The system includes a FastAPI backend for inference and a Streamlit frontend for interaction and visualization.

---

## Live Demo

Frontend (UI): https://credit-risk-ui.onrender.com
API Docs: https://credit-risk-api-scq0.onrender.com/docs

---

## Problem Statement

Machine learning models degrade silently when real-world data changes. A model trained on past economic conditions may produce unreliable predictions under new conditions. This project addresses that by combining prediction with automated drift detection.

---

## System Architecture

```
User (Streamlit UI)
        ↓
FastAPI Backend (Render)
        ↓
Preprocessing + ML Model
        ↓
Prediction + Risk Classification
        ↓
Drift Detection Monitoring
```

---

## Features

### Risk Prediction

* Predicts loan default probability
* Classifies risk into:

  * Low
  * Medium
  * High
  * Very High

### Interactive UI

* Built using Streamlit
* Multi-page interface (Dashboard, Prediction, Batch, Drift Monitoring)
* Real-time feedback and metrics

### Batch Prediction

* Supports multiple input records
* Scalable prediction pipeline

### Drift Detection

* Uses Population Stability Index (PSI)
* Detects distribution shifts in incoming data
* Provides actionable alerts (Stable, Warning, Critical)

---

## Dataset

Credit Risk Dataset (32,581 records, 12 features)
Source: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Handled real-world issues:

* Outliers (extreme ages and employment values)
* Missing values in key fields
* Class imbalance (78% non-default, 22% default)

---

## Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 78.8%    | 51.1%     | 77.6%  | 61.6%    | 85.4%   |
| Random Forest       | 91.8%    | 86.9%     | 73.6%  | 79.7%    | 92.8%   |
| Gradient Boosting   | 94.2%    | 97.1%     | 75.6%  | 85.0%    | 95.5%   |
| XGBoost             | 92.9%    | 86.2%     | 80.6%  | 83.3%    | 95.3%   |

Best model: Gradient Boosting

---

## Tech Stack

* Python 3.11
* Scikit-learn, XGBoost, LightGBM
* FastAPI (Backend API)
* Streamlit (Frontend UI)
* Pandas, NumPy
* Render (Deployment)

---

## Project Structure

```
credit-risk-drift-detection/
├── app.py                     # Streamlit entry point
├── pages/                    # UI pages
├── api/                      # FastAPI backend
├── models/                   # Trained model + preprocessor
├── src/                      # Training + preprocessing
├── utils/                    # API helper functions
├── requirements.txt          # UI dependencies
├── requirements-backend.txt  # Backend dependencies
└── README.md
```

---

## Running Locally

### Backend

```
pip install -r requirements-backend.txt
uvicorn api.app:app --reload
```

### Frontend

```
pip install -r requirements.txt
streamlit run app.py
```

---

## API Example

```
POST /predict
```

Request:

```
{
  "person_age": 30,
  "person_income": 60000,
  "person_home_ownership": "MORTGAGE",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 8
}
```

Response:

```
{
  "prediction": "FULLY_PAID",
  "default_probability": 0.0312,
  "risk_level": "LOW",
  "confidence": 0.9688
}
```

---

## Drift Detection

Population Stability Index (PSI):

```
PSI = Σ (current% - reference%) × ln(current% / reference%)
```

| PSI Value | Interpretation    | Action           |
| --------- | ----------------- | ---------------- |
| < 0.1     | Stable            | No action needed |
| 0.1–0.2   | Moderate drift    | Investigate      |
| > 0.2     | Significant drift | Retrain model    |

---

## Notes

* Backend may take 20–30 seconds to respond on first request (Render free tier)
* Retry logic is implemented in the frontend to handle this

---

## Future Improvements

* SHAP-based explainability
* Automated retraining pipeline
* Real-time monitoring dashboard
* Authentication and user roles
* Model versioning and tracking

---


