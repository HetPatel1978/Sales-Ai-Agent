# 🤖 AI-Powered Sales Agent

A complete end-to-end AI agent that predicts the success probability of sales deals, recommends the next best action, and generates personalized follow-up emails using Hugging Face Inference API.

---

## 🚀 Features

- ✅ **Deal Success Prediction**: ML model trained on CRM-style sales data to classify potential deal outcomes.
- ✅ **Action Recommendation**: Rule-based engine suggests the next best action based on probability and recency.
- ✅ **AI Email Generator**: Uses Hugging Face's LLMs to auto-generate personalized follow-up emails.
- ✅ **Interactive Dashboard**: Built with Streamlit for fast prototyping and user interaction.
- ✅ **Modular API**: FastAPI backend (optional phase) to decouple UI from logic.

---

## 📊 Tech Stack

| Layer | Tech |
|------|------|
| ML Model | XGBoost, Scikit-Learn |
| Dashboard | Streamlit |
| Backend | Python, FastAPI |
| Email AI | Hugging Face Inference API |
| Hosting | Hugging Face Spaces / Streamlit Cloud |

---

## 📂 Folder Structure

```
sales_ai_agent/
├── data/                   # Raw and processed datasets
├── models/                 # Trained model, encoders, scaler
├── src/
│   ├── dashboard_app.py    # Streamlit main app
│   ├── data_preprocessing.py
│   ├── deal_success_model.py
│   ├── action_recommender.py
│   ├── email_generator.py
│   └── api.py              # (optional) FastAPI app
├── config.yaml             # App configuration
├── requirements.txt
├── app.py                  # Entry point for Hugging Face deployment
├── LICENSE
└── README.md
```

---

## 🧠 How It Works

1. User fills lead details in dashboard
2. Trained model predicts deal success probability
3. Rule engine recommends action (e.g., "Arrange Demo")
4. LLM generates follow-up email.

---

## 🛠 How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/HetPatel1978/Sales-Ai-Agent
cd Sales-Ai-Agent
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run src/dashboard_app.py
```


## 📄 License

This project is under **All Rights Reserved**. No part of this repository may be used, copied, or modified without explicit permission from the author.

---

## 🙌 Acknowledgements
- Hugging Face Inference API
- Scikit-learn, XGBoost
- Streamlit & FastAPI teams

---

## ✨ Created by HET PATEL

