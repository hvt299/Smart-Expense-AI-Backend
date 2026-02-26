from fastapi import FastAPI
from pydantic import BaseModel
from app.ml_model.dummy_ai import predict_expense_category

app = FastAPI(title="Smart Expense AI API")

class ExpenseRequest(BaseModel):
    text: str

class ExpenseResponse(BaseModel):
    text: str
    category: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Smart Expense AI API!"}

@app.post("/predict", response_model=ExpenseResponse)
def predict_expense(request: ExpenseRequest):
    predicted_category = predict_expense_category(request.text)
    
    return ExpenseResponse(
        text=request.text,
        category=predicted_category
    )