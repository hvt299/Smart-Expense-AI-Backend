import os
import re
import joblib
from app.ml_model.text_utils import tokenize_vn, clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "expense_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("Model chưa được train")

model = joblib.load(model_path)

def extract_amount(text: str) -> float:
    text = text.lower()

    multipliers = {
        "k": 1000, "ngàn": 1000, "nghìn": 1000,
        "tr": 1000000, "triệu": 1000000, "củ": 1000000,
        "tỷ": 1000000000
    }

    pattern = r"(\d+(?:\.\d+)?)\s*(k|ngàn|nghìn|tr|triệu|củ|tỷ)?"
    matches = re.findall(pattern, text)

    amounts = []
    for num, unit in matches:
        amount = float(num)
        if unit in multipliers:
            amount *= multipliers[unit]
        elif amount < 1000:
            amount *= 1000
        amounts.append(amount)

    return max(amounts) if amounts else 0.0

def analyze_expense(text: str) -> dict:
    if not text or not text.strip():
        return {
            "amount": 0.0,
            "category": "unknown",
            "note": ""
        }

    amount = extract_amount(text)
    clean = clean_text(text)
    category = model.predict([clean])[0]

    return {
        "amount": amount,
        "category": str(category),
        "note": text.strip()
    }