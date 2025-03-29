# src/data_preprocessing.py

"""
Phase 1 - Data Preprocessing Script
Task:
- Encode Categorical Variables
- Scale Numerical Features
- Save Preprocessed Data
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------- Load Converted Data ----------
def load_data():
    df = pd.read_csv(r"C:/Users/hetpa/OneDrive/Desktop/AIML/AI Agent Sales/Data/processes data/converted_sales_data.csv")
    print(f"Loaded {len(df)} records from converted_sales_data.csv")
    return df

# ---------- Encode + Scale ----------
# def preprocess_data(df):
#     encoders = {}

#     # Encode Categorical Variables
#     for col in ['Industry', 'Sales Stage', 'Lead Source']:
#         encoder = LabelEncoder()
#         df[col] = encoder.fit_transform(df[col])
#         encoders[col] = encoder
#     print("Categorical Encoding Completed.")

#     # Scale Numerical Features
#     scaler = StandardScaler()
#     df[['Deal Amount', 'Emails', 'Meetings']] = scaler.fit_transform(df[['Deal Amount', 'Emails', 'Meetings']])
#     print("Feature Scaling Completed.")

#     return df, encoders, scaler


def preprocess_data(df):
    encoders = {}

    # --- Fix date ---
    df['Last Contact Date'] = pd.to_datetime(df['Last Contact Date'])
    df['Days Since Last Contact'] = (pd.Timestamp.today() - df['Last Contact Date']).dt.days
    df = df.drop(columns=['Last Contact Date'])

    # --- Encode categoricals ---
    for col in ['Industry', 'Sales Stage', 'Lead Source']:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    print("Categorical Encoding Completed.")

    # --- Scale numerical ---
    scaler = StandardScaler()
    df[['Deal Amount', 'Emails', 'Meetings', 'Days Since Last Contact']] = scaler.fit_transform(
        df[['Deal Amount', 'Emails', 'Meetings', 'Days Since Last Contact']]
    )

    print("Feature Scaling Completed.")
    return df, encoders, scaler


# ---------- Save Preprocessed Data + Encoders ----------
def save_outputs(df, encoders, scaler):
    os.makedirs("../data/processes data", exist_ok=True)
    os.makedirs("../Models", exist_ok=True)

    df.to_csv("../data/processes data/preprocessed_sales_data.csv", index=False)
    joblib.dump(encoders, "../Models/encoders.pkl")
    joblib.dump(scaler, "../Models/scaler.pkl")

    print("Preprocessed data, encoders, and scaler saved successfully.")

# ---------- Main ----------
if __name__ == "__main__":
    df = load_data()
    df, encoders, scaler = preprocess_data(df)
    save_outputs(df, encoders, scaler)
