# # src/email_generator.py

# """
# Phase 4 - AI Email Generator (OpenAI API >= 1.x compatible)
# """

# import openai
# import yaml

# # ---------- Load API Key ----------
# with open("../config.yaml", "r") as file:
#     config = yaml.safe_load(file)

# openai.api_key = config['openai_api_key']

# # ---------- Email Generator ----------
# def generate_email(lead_info: dict, recommended_action: str) -> str:
#     """
#     Generates a follow-up email using OpenAI ChatCompletion
#     """

#     prompt = f"""
#     You are a professional sales agent.
#     Write a short and professional follow-up email.

#     Lead Information:
#     - Industry: {lead_info['Industry']}
#     - Sales Stage: {lead_info['Sales Stage']}
#     - Deal Amount: ${lead_info['Deal Amount']}
#     - Recommended Action: {recommended_action}

#     Provide a concise and effective email.
#     """

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7
#     )

#     email_text = response['choices'][0]['message']['content'].strip()
#     return email_text

















# # src/email_generator.py

# """
# Phase 4 - Email Generator using OpenRouter official client
# """

# from openai import OpenAI
# import yaml

# # ---------- Load API Key ----------
# with open("../config.yaml", "r") as file:
#     config = yaml.safe_load(file)

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=config['openai_api_key']
# )

# # ---------- Email Generator ----------
# def generate_email(lead_info: dict, recommended_action: str) -> str:
#     """
#     Generates a follow-up email using OpenRouter Chat API
#     """

#     prompt = f"""
#     You are a professional sales agent.
#     Write a short and professional follow-up email.

#     Lead Information:
#     - Industry: {lead_info['Industry']}
#     - Sales Stage: {lead_info['Sales Stage']}
#     - Deal Amount: ${lead_info['Deal Amount']}
#     - Recommended Action: {recommended_action}

#     Provide a concise and effective email.
#     """

#     completion = client.chat.completions.create(
#         extra_headers={
#             "HTTP-Referer": "http://localhost",    # Optional: update when deployed
#             "X-Title": "Sales AI Agent"
#         },
#         model="openai/gpt-4o",  # You can change this to any OpenRouter supported model
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )

#     return completion.choices[0].message.content.strip()







# src/email_generator.py

"""
Phase 4 - Email Generator using Hugging Face Inference API
"""

import requests
import yaml

# ---------- Load Config ----------
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

api_url = f"https://api-inference.huggingface.co/models/{config['model_name']}"
headers = {"Authorization": f"Bearer {config['huggingface_api_key']}"}

# ---------- Email Generator ----------
def generate_email(lead_info: dict, recommended_action: str) -> str:
    """
    Generates a follow-up email using Hugging Face model
    """

    prompt = f"""
    You are a professional sales agent.
    Write a short and professional follow-up email.

    Lead Information:
    - Industry: {lead_info['Industry']}
    - Sales Stage: {lead_info['Sales Stage']}
    - Deal Amount: ${lead_info['Deal Amount']}
    - Recommended Action: {recommended_action}

    Provide a concise and effective email.
    """

    response = requests.post(api_url, headers=headers, json={"inputs": prompt})
    
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    generated_text = response.json()[0]['generated_text']
    return generated_text.strip()

