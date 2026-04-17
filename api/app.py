import os
import sys
import json
import pickle
import joblib
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

# Fix import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML-powered loan default risk assessment with drift monitoring",
    version="1.0.0"
)

# ✅ CORS FIX (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

model = None
preprocessor = None
metadata = None


# -------------------------------
# Custom Unpickler
# -------------------------------
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "CreditRiskPreprocessor":
            from data_preprocessing import CreditRiskPreprocessor
            return CreditRiskPreprocessor
        return super().find_class(module, name)


# -------------------------------
# Load model
# -------------------------------
def load_artifacts():
    global model, preprocessor, metadata

    model = joblib.load(MODEL_PATH)

    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception:
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = CustomUnpickler(f).load()

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)


# -------------------------------
# Schemas
# -------------------------------
class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: int = Field(..., gt=0)
    person_home_ownership: str
    person_emp_length: Optional[float] = Field(None, ge=0, le=60)
    loan_intent: str
    loan_grade: str
    loan_amnt: int = Field(..., gt=0)
    loan_int_rate: Optional[float] = Field(None, gt=0, le=25)
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: str
    default_probability: float
    risk_level: str
    confidence: float
    timestamp: str


class BatchRequest(BaseModel):
    applications: List[LoanApplication]


# -------------------------------
# Startup
# -------------------------------
@app.on_event("startup")
async def startup():
    try:
        load_artifacts()
        print("Model loaded successfully")
    except Exception as e:
        print("Startup error:", e)
        raise


# -------------------------------
# Routes
# -------------------------------

# ✅ Root route (prevents 404 confusion)
@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


# ✅ FIXED: match frontend naming
@app.get("/model-info")
def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
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


# ✅ FIXED: match frontend naming
@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    for app_data in request.applications:
        input_df = pd.DataFrame([app_data.dict()])
        X = preprocessor.transform(input_df, has_target=False)

        prob = float(model.predict_proba(X)[0][1])

        results.append({
            "default_probability": round(prob, 4),
            "prediction": "DEFAULT" if prob >= 0.5 else "FULLY_PAID"
        })

    return {
        "predictions": results,
        "count": len(results)
    }


# -------------------------------
# Run locally
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)