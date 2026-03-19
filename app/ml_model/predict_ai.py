import os
import re
import joblib
from app.ml_model.text_utils import tokenize_vn, clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "expense_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("Model chưa được train")

models = joblib.load(model_path)
cat_model = models["category_model"]
type_model = models["type_model"]

def extract_amount(text: str) -> float:
    text = text.lower()
    text = re.sub(r'(\d+)(triệu|tr|nghìn|ngàn|k|củ|tỷ|lít)(\d+)', r'\1.\3\2', text)
    text = re.sub(r'(vnđ|đ|vnd)\b', '', text).strip()

    multipliers = {
        "k": 1000, "ngàn": 1000, "nghìn": 1000,
        "tr": 1000000, "triệu": 1000000, "củ": 1000000,
        "tỷ": 1000000000, "lít": 100000
    }

    pattern = r"(\d+(?:[\.,]\d+)*)\s*(triệu|tr|nghìn|ngàn|k|củ|tỷ|lít)?"
    matches = re.findall(pattern, text)

    amounts = []
    for num_str, unit in matches:
        num_str = num_str.replace(',', '.')

        if num_str.count('.') > 1:
            num_str = num_str.replace('.', '')
        elif num_str.count('.') == 1:
            parts = num_str.split('.')
            if len(parts[1]) == 3:
                num_str = num_str.replace('.', '')
                
        amount = float(num_str)

        if unit in multipliers:
            amount *= multipliers[unit]
        elif not re.search(r"[\.,]", num_str) and amount < 1000 and amount > 0:
            amount *= 1000
            
        amounts.append(amount)

    return max(amounts) if amounts else 0.0

def analyze_expense(text: str) -> dict:
    if not text or not text.strip():
        return {
            "amount": 0.0,
            "category": "Khác",
            "type": "expense",
            "note": ""
        }
    
    amount = extract_amount(text)
    cleaned = clean_text(text)
    
    category = cat_model.predict([cleaned])[0]
    txn_type = type_model.predict([cleaned])[0]
    
    note_pattern = r"\b\d+(?:[\.,]\d+)?\s*(triệu|tr|nghìn|ngàn|k|củ|tỷ|lít|vnđ|đ|vnd)\b"
    note = re.sub(note_pattern, "", text.lower()).strip()
    
    return {
        "amount": amount,
        "category": category,
        "type": txn_type,
        "note": note.capitalize() if note else text.capitalize()
    }