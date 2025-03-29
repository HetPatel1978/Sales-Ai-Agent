# src/deal_success_model.py

"""
Phase 2 - Deal Success Prediction Model
Includes:
- Model Training
- Evaluation
- Model Saving
"""

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import xgboost as xgb

# ---------- Load Preprocessed Data ----------
def load_data():
    df = pd.read_csv(r"C:/Users/hetpa/OneDrive/Desktop/AIML/AI Agent Sales/Data/processes data/preprocessed_sales_data.csv")
    print(f"Loaded {len(df)} records for training.")
    return df

# ---------- Model Training ----------
def train_model(df):
    X = df.drop('Deal Status', axis=1)
    y = df['Deal Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        scale_pos_weight=5,  # handle imbalance
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # ---------- Evaluation ----------
    y_pred = model.predict(X_test)

    print("\n===== Model Evaluation =====")
    print(f"AUC Score    : {roc_auc_score(y_test, y_pred):.2f}")
    print(f"Precision    : {precision_score(y_test, y_pred):.2f}")
    print(f"Recall       : {recall_score(y_test, y_pred):.2f}")
    print("============================\n")

    # ---------- Save Model ----------
    os.makedirs("../Models", exist_ok=True)
    joblib.dump(model, "../Models/deal_success_model.pkl")
    print("Model saved at: ../Models/deal_success_model.pkl")

    return model

# ---------- Main ----------
if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
