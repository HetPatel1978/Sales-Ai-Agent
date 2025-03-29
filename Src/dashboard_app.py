# src/dashboard_app.py

"""
Phase 6 - Streamlit Dashboard for Sales AI Agent (Clean Version)
"""

import streamlit as st
import pandas as pd
import joblib
import yaml

from action_recommender import recommend_action
from email_generator import generate_email

# ---------- Load Configuration & Assets ----------
config = yaml.safe_load(open("../config.yaml"))
model = joblib.load("../models/deal_success_model.pkl")
encoders = joblib.load("../models/encoders.pkl")
scaler = joblib.load("../models/scaler.pkl")

# ---------- Streamlit Page Layout ----------
st.set_page_config(page_title="Sales AI Agent", layout="centered")
st.title("🚀 Sales AI Agent Dashboard")
st.markdown("Enter lead details below to get predicted deal success, a recommended next step, and an AI-generated follow-up email.")

# ---------- Lead Input Form ----------
with st.form("lead_form"):
    industry = st.text_input("Industry", placeholder="e.g., Healthcare")
    sales_stage = st.selectbox("Sales Stage", ["Contacted", "Negotiation", "Proposal Sent"])
    lead_source = st.selectbox("Lead Source", ["Website", "Referral", "Social Media", "Event"])
    deal_amount = st.number_input("Deal Amount (USD)", min_value=0.0)
    emails = st.number_input("Number of Emails", min_value=0)
    meetings = st.number_input("Number of Meetings", min_value=0)
    days_since_last_contact = st.number_input("Days Since Last Contact", min_value=0)

    submitted = st.form_submit_button("Predict Now")

# ---------- Main Pipeline ----------
if submitted:
    # 1. Construct Input DataFrame with Original Column Names
    lead_df = pd.DataFrame([{
        "Industry": industry,
        "Sales Stage": sales_stage,
        "Lead Source": lead_source,
        "Deal Amount": deal_amount,
        "Emails": emails,
        "Meetings": meetings,
        "Days Since Last Contact": days_since_last_contact
    }])

    try:
        # 2. Encode Categorical Columns
        for col in ['Industry', 'Sales Stage', 'Lead Source']:
            lead_df[col] = encoders[col].transform(lead_df[col])

        # 3. Scale Numeric Features
        lead_df[['Deal Amount', 'Emails', 'Meetings', 'Days Since Last Contact']] = scaler.transform(
            lead_df[['Deal Amount', 'Emails', 'Meetings', 'Days Since Last Contact']]
        )

        # 4. Predict Probability
        probability = model.predict_proba(lead_df)[0][1]
        success_percent = round(probability * 100, 2)

        # 5. Recommend Next Action
        recommendation = recommend_action(probability, days_since_last_contact)

        # 6. Generate Follow-Up Email
        email = generate_email({
            "Industry": industry,
            "Sales Stage": sales_stage,
            "Deal Amount": deal_amount
        }, recommendation)

        # ---------- Display Results ----------
        st.subheader("📊 Prediction Results")
        st.write(f"**Success Probability:** {success_percent}%")
        st.write(f"**Recommended Action:** {recommendation}")

        st.subheader("✉ AI-Generated Follow-up Email")
        st.text_area("Generated Email", email, height=200)

    except KeyError as e:
        st.error(f"Error: {str(e)}\n\nPlease make sure you used a known category for: {e}")
    except Exception as ex:
        st.error(f"Unexpected Error: {ex}")

# ---------- Disclaimer ----------
st.caption("""
📌 **Disclaimer:** This AI Agent is trained on a public retail dataset with synthetic features and labels. 
It is intended for educational and demonstration purposes only.
""")
