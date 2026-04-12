# Credit Risk Prediction with Automated Drift Detection

A production-grade ML pipeline that predicts loan defaults using real credit risk data and **automatically detects when the model becomes unreliable** due to data drift — ensuring predictions stay trustworthy over time.

## Why This Matters

ML models degrade silently. A loan default model trained on pre-recession data will give dangerously wrong predictions during an economic downturn — approving risky borrowers and rejecting safe ones. This project solves that by monitoring incoming data distributions and raising alerts when they deviate from training data.

## Architecture

```
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
```

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
| **Gradient Boosting** | **93.9%** | **96.5%** | **75.0%** | **84.4%** | **95.6%** |
| XGBoost | 92.9% | 86.2% | 80.6% | 83.3% | 95.3% |

**Best model**: Gradient Boosting with 93.9% accuracy and 95.6% ROC AUC

### Drift Detection Results

| Scenario | Drifted Features | Severity | Action |
|----------|-----------------|----------|--------|
| Control (same data) | 0/11 (0%) | STABLE | None needed |
| Mild downturn | 1/11 (9%) | WARNING | Investigate |
| Severe downturn | 4/11 (36%) | CRITICAL | Retrain immediately |

### Top Risk Factors (Feature Importance)

1. **Income-to-Loan Ratio** (27.2%) — Most predictive feature
2. **Loan Grade** (20.5%) — Assigned risk grade matters
3. **Home Ownership** (16.7%) — Stability indicator
4. **Person Income** (14.2%) — Ability to repay
5. **Loan Intent** (8.5%) — Purpose affects risk

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Models | Scikit-learn, XGBoost |
| Drift Detection | Custom PSI + KS tests (SciPy) |
| API | FastAPI + Uvicorn |
| Data Processing | Pandas, NumPy |

## Project Structure

```
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
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
python src/data_preprocessing.py
```
Cleans data (removes 144 outlier rows), engineers 5 new features, handles missing values, splits into train/test.

### 3. Train Models
```bash
python src/train.py
```
Trains 4 models, compares all metrics, saves the best (Gradient Boosting).

### 4. Run Drift Detection
```bash
python src/drift_detector.py
```
Runs 3 tests: control (no drift), mild downturn, severe downturn. Outputs per-feature PSI scores and overall severity assessment.

### 5. Start Prediction API
```bash
cd api && uvicorn app:app --reload
```
Visit `http://localhost:8000/docs` for interactive Swagger UI.

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
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
```

Response:
```json
{
  "prediction": "FULLY_PAID",
  "default_probability": 0.0312,
  "risk_level": "LOW",
  "confidence": 0.9688
}
```

## How Drift Detection Works

**Population Stability Index (PSI)** measures how much a feature's distribution has shifted:

```
PSI = Σ (current% - reference%) × ln(current% / reference%)
```

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | Model is fine |
| 0.1 - 0.2 | Moderate shift | Investigate the cause |
| > 0.2 | Significant drift | Retrain the model |

The system also uses **Kolmogorov-Smirnov tests** for statistical validation and **categorical distribution comparison** for non-numeric features.

## Key Design Decisions

1. **F1 over Accuracy**: With 22% default rate, a naive model predicting "no default" always gets 78% accuracy. F1 balances precision (don't flag good borrowers) and recall (don't miss actual defaults).

2. **Feature Engineering**: Created `income_to_loan` ratio which became the #1 most important feature — raw income and loan amount alone are less predictive than their ratio.

3. **Outlier Removal**: Ages of 144 and employment lengths of 123 years are clearly data errors. Removing them (only 0.4% of data) improved model reliability.

4. **Simulated Drift**: Since the dataset lacks timestamps, drift is simulated via realistic economic scenarios (income drops, rate increases). In production, you'd compare by time windows.

## Future Improvements

- [ ] Streamlit dashboard for real-time drift visualization
- [ ] Automated retraining pipeline triggered on drift detection
- [ ] MLflow experiment tracking integration
- [ ] Docker containerization for deployment
- [ ] SHAP explainability for individual predictions
- [ ] Unit tests with pytest

## License

MIT
