cat > README.md << 'EOF'
# Credit Risk Prediction with Automated Drift Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

A production-grade ML pipeline that predicts loan defaults using real credit risk data and **automatically detects when the model becomes unreliable** due to data drift — ensuring predictions stay trustworthy over time.

## 🚀 Live Demo
**API Docs**: https://credit-risk-api-scq0.onrender.com/docs


## Why This Matters

ML models degrade silently. A loan default model trained on pre-recession data will give dangerously wrong predictions during an economic downturn — approving risky borrowers and rejecting safe ones. This project solves that by monitoring incoming data distributions and raising alerts when they deviate from training data.

## Architecture

\`\`\`
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Credit Risk     │────▶│  Preprocessing   │────▶│  Model Training │
│  Dataset (32K)   │     │  + Feature Eng   │     │  (4 models)     │
└──────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Drift           │◀────│  New Production  │────▶│  FastAPI        │
│  Detector (PSI)  │     │  Data            │     │  Prediction API │
└──────┬───────────┘     └──────────────────┘     └─────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  STABLE / WARNING / CRITICAL - RETRAIN NOW  │
└─────────────────────────────────────────────┘
\`\`\`

## Dataset

**Credit Risk Dataset** from Kaggle (32,581 real loan records, 12 features)
Source: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Real-world data issues handled:
- **Outliers**: Ages up to 144, employment length up to 123 years
- **Missing values**: 895 in employment length, 3,116 in interest rate
- **Class imbalance**: 78% paid vs 22% default

## Results

### Model Performance (on real test data)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 78.8% | 51.1% | 77.6% | 61.6% | 85.4% |
| Random Forest | 91.8% | 86.9% | 73.6% | 79.7% | 92.8% |
| **Gradient Boosting** | **94.2%** | **97.1%** | **75.6%** | **85.0%** | **95.5%** |
| XGBoost | 92.9% | 86.2% | 80.6% | 83.3% | 95.3% |

**Best model**: Gradient Boosting with 94.2% accuracy and 95.5% ROC AUC

### Drift Detection Results

| Scenario | Drifted Features | Severity | Action |
|----------|-----------------|----------|--------|
| Control (same data) | 0/11 (0%) | STABLE | None needed |
| Mild downturn | 1/11 (9%) | WARNING | Investigate |
| Severe downturn | 4/11 (36%) | CRITICAL | Retrain immediately |

### Top Risk Factors (Feature Importance)

1. **Income-to-Loan Ratio** (27.0%) — Most predictive feature
2. **Loan Grade** (20.4%) — Assigned risk grade matters
3. **Home Ownership** (16.7%) — Stability indicator
4. **Person Income** (14.2%) — Ability to repay
5. **Loan Intent** (8.5%) — Purpose affects risk

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Models | Scikit-learn, XGBoost |
| Drift Detection | Custom PSI + KS tests (SciPy) |
| API | FastAPI + Uvicorn |
| Data Processing | Pandas, NumPy |
| Deployment | Render |

## Project Structure

\`\`\`
credit-risk-drift-detection/
├── data/
│   ├── raw/credit_risk_dataset.csv    # Real Kaggle dataset
│   └── processed/                     # Train/test splits + reference data
├── src/
│   ├── data_preprocessing.py          # Outlier removal, feature engineering, encoding
│   ├── train.py                       # Multi-model training & comparison
│   └── drift_detector.py              # PSI & KS drift monitoring
├── api/
│   └── app.py                         # FastAPI prediction service
├── models/                            # Saved model + preprocessor + metadata
├── reports/                           # Drift detection JSON reports
├── requirements.txt
└── README.md
\`\`\`

## Quick Start

> **Prerequisites**: Python 3.11, create a virtual environment first:
> \`python3.11 -m venv venv311 && source venv311/bin/activate\`

### 1. Install Dependencies
\`\`\`bash
pip install -r requirements.txt && pip install lightgbm==4.6.0
\`\`\`

### 2. Run Preprocessing
\`\`\`bash
python3 -c "import sys; sys.path.insert(0, 'src'); from data_preprocessing import run_preprocessing; run_preprocessing()"
\`\`\`

### 3. Train Models
\`\`\`bash
python src/train.py
\`\`\`

### 4. Run Drift Detection
\`\`\`bash
python src/drift_detector.py
\`\`\`

### 5. Start Prediction API
\`\`\`bash
PYTHONPATH=. uvicorn api.app:app --reload
\`\`\`
Visit \`http://localhost:8000/docs\` for interactive Swagger UI.

## API Usage

\`\`\`bash
curl -X POST https://credit-risk-api-scq0.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
\`\`\`

Response:
\`\`\`json
{
  "prediction": "FULLY_PAID",
  "default_probability": 0.0312,
  "risk_level": "LOW",
  "confidence": 0.9688
}
\`\`\`

## How Drift Detection Works

**Population Stability Index (PSI)** measures how much a feature's distribution has shifted:

\`\`\`
PSI = Σ (current% - reference%) × ln(current% / reference%)
\`\`\`

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | Model is fine |
| 0.1 - 0.2 | Moderate shift | Investigate the cause |
| > 0.2 | Significant drift | Retrain the model |

## Key Design Decisions

1. **F1 over Accuracy**: With 22% default rate, a naive model predicting "no default" always gets 78% accuracy. F1 balances precision and recall.

2. **Feature Engineering**: Created \`income_to_loan\` ratio which became the #1 most important feature.

3. **Outlier Removal**: Ages of 144 and employment lengths of 123 years are data errors. Removing them improved model reliability.

4. **Module-based Preprocessing**: Preprocessor is saved with correct module reference to ensure deployment compatibility.

## Future Improvements

- [ ] Streamlit dashboard for real-time drift visualization
- [ ] Automated retraining pipeline triggered on drift detection
- [ ] MLflow experiment tracking integration
- [ ] Docker containerization for deployment
- [ ] SHAP explainability for individual predictions
- [ ] Unit tests with pytest

## License

MIT
EOF
