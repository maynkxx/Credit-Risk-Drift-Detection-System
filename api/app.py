"""
FastAPI Prediction Service for Credit Risk Model.

Endpoints:
- POST /predict       -> Single loan prediction
- POST /predict/batch -> Batch predictions
- GET  /health        -> Health check
- GET  /model/info    -> Model metadata
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import CreditRiskPreprocessor  # noqa

from data_preprocessing import CreditRiskPreprocessor  # noqa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML-powered loan default risk assessment with drift monitoring",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

model = None
preprocessor = None
metadata = None


def load_artifacts():
    global model, preprocessor, metadata
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)


class LoanApplication(BaseModel):
    """Input schema matching the Credit Risk Dataset columns."""
    person_age: int = Field(..., ge=18, le=100, description="Applicant age")
    person_income: int = Field(..., gt=0, description="Annual income")
    person_home_ownership: str = Field(..., description="RENT, OWN, MORTGAGE, or OTHER")
    person_emp_length: Optional[float] = Field(None, ge=0, le=60, description="Employment length (years)")
    loan_intent: str = Field(..., description="PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="Loan grade A-G")
    loan_amnt: int = Field(..., gt=0, description="Loan amount")
    loan_int_rate: Optional[float] = Field(None, gt=0, le=25, description="Interest rate %")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan as % of income")
    cb_person_default_on_file: str = Field(..., description="Historical default: Y or N")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length (years)")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    prediction: str
    default_probability: float
    risk_level: str
    confidence: float
    timestamp: str


class BatchRequest(BaseModel):
    applications: List[LoanApplication]


@app.on_event("startup")
async def startup():
    try:
        load_artifacts()
        print("Model and preprocessor loaded successfully!")
    except Exception as e:
        import traceback
        print("STARTUP ERROR:", e)
        traceback.print_exc()
        raise


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return metadata


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """Predict default risk for a single loan application."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([application.dict()])
    X = preprocessor.transform(input_df, has_target=False)

    probability = float(model.predict_proba(X)[0][1])
    prediction = "DEFAULT" if probability >= 0.5 else "FULLY_PAID"

    if probability < 0.2:
        risk_level = "LOW"
    elif probability < 0.4:
        risk_level = "MODERATE"
    elif probability < 0.6:
        risk_level = "HIGH"
    else:
        risk_level = "VERY_HIGH"

    return PredictionResponse(
        prediction=prediction,
        default_probability=round(probability, 4),
        risk_level=risk_level,
        confidence=round(max(probability, 1 - probability), 4),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """Predict default risk for multiple applications."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for app_data in request.applications:
        input_df = pd.DataFrame([app_data.dict()])
        X = preprocessor.transform(input_df, has_target=False)
        prob = float(model.predict_proba(X)[0][1])

        results.append({
            "default_probability": round(prob, 4),
            "prediction": "DEFAULT" if prob >= 0.5 else "FULLY_PAID",
        })

    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
