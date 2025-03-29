# src/action_recommender.py

"""
Phase 3 - Rule-based Action Recommender

This module recommends the next action based on:
- Success Probability from the model
- Days Since Last Contact
"""

import yaml

# Load thresholds from config.yaml
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

def recommend_action(probability: float, days_since_last_contact: int) -> str:
    """
    Rule-based recommender:
    - High probability + recent contact → Arrange Demo
    - High probability + old contact → Immediate Follow-up
    - Medium probability → Follow-up + Case Study
    - Low probability → Wait or Disqualify
    """
    high = config['recommendation_thresholds']['high']
    medium = config['recommendation_thresholds']['medium']
    overdue_days = config['recommendation_thresholds']['overdue_days']

    if probability > high:
        if days_since_last_contact > overdue_days:
            return "Immediate Follow-up"
        else:
            return "Arrange Demo"
    elif medium < probability <= high:
        return "Follow-up + Case Study"
    else:
        return "Wait or Disqualify"








#for refrence 
# # src/test_recommender.py

# from action_recommender import recommend_action

# # Example
# probability = 0.75
# days_since_last_contact = 5

# recommendation = recommend_action(probability, days_since_last_contact)
# print("Recommended Action:", recommendation)
