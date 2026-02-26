import random

def predict_expense_category(text: str) -> str:
    categories = ["food_beverage", "transportation", "shopping", "housing_bills"]
    return random.choice(categories)