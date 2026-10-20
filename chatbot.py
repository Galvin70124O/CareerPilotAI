"""
Command-line version of CareerPilot AI. Uses the same model as the web app.

    python train.py      # once, creates model.pkl
    python chatbot.py
"""
import os

import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model file not found. Run 'python train.py' first.")
model = joblib.load(MODEL_PATH)

while True:
    user_input = input("\nEnter your skills or resume text (type 'exit' to quit):\n")
    if user_input.lower() == "exit":
        break
    probabilities = model.predict_proba([user_input])[0]
    top3 = sorted(zip(model.classes_, probabilities), key=lambda pair: pair[1], reverse=True)[:3]
    print(f"Recommended Career Path: {top3[0][0]}")
    for career, prob in top3:
        print(f"  {career:<28}{prob:.0%}")
