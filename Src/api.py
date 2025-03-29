# src/api.py

"""
Phase 5 - FastAPI Application for Sales AI Agent (Fixed Version)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import yaml

from action_recommender import recommend_action
from email_generator import generate_email

# ---------- Load Config, Model, Encoder & Scaler ----------
config = yaml.safe_load(open("../config.yaml"))

model = joblib.load("../Models/deal_success_model.pkl")
encoders = joblib.load("../Models/encoders.pkl")
scaler = joblib.load("../Models/scaler.pkl")

# ---------- FastAPI App ----------
app = FastAPI(title="Sales AI Agent", description="Deal Prediction, Recommendation, and Email Generator")

# ---------- Define Lead Input Schema ----------
class Lead(BaseModel):
    Industry: str
    Sales_Stage: str   # ✅ Use underscore for API input
    Lead_Source: str   # ✅ Use underscore for API input
    Deal_Amount: float
    Emails: int
    Meetings: int
    Days_Since_Last_Contact: int

# ---------- Preprocess Input ----------
def preprocess_input(lead: Lead) -> pd.DataFrame:
    data = pd.DataFrame([lead.dict()])

    # ✅ Rename API-friendly keys to match model-friendly keys
    data.rename(columns={
        "Sales_Stage": "Sales Stage",
        "Lead_Source": "Lead Source",
        "Deal_Amount": "Deal Amount",
        "Days_Since_Last_Contact": "Days Since Last Contact"
    }, inplace=True)

    # Encode
    for col in ['Industry', 'Sales Stage', 'Lead Source']:
        data[col] = encoders[col].transform(data[col])

    # Scale
    data[['Deal Amount', 'Emails', 'Meetings', 'Days Since Last Contact']] = scaler.transform(
        data[['Deal Amount', 'Emails', 'Meetings', 'Days Since Last Contact']]
    )

    return data

# ---------- API Route ----------
@app.post("/predict")
def predict(lead: Lead):
    # Preprocess
    data = preprocess_input(lead)

    # Predict
    probability = model.predict_proba(data)[0][1]

    # Recommend Action
    recommendation = recommend_action(probability, lead.Days_Since_Last_Contact)

    # Generate Email
    email = generate_email({
        "Industry": lead.Industry,
        "Sales Stage": lead.Sales_Stage,
        "Deal Amount": lead.Deal_Amount
    }, recommendation)

    return {
        "success_probability": round(float(probability), 3),
        "recommended_action": recommendation,
        "generated_email": email
    }
