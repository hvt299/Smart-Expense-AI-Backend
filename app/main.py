import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.ml_model.predict_ai import analyze_expense

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Smart Expense AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExpenseRequest(BaseModel):
    text: str

class ExpenseResponse(BaseModel):
    amount: float
    category: str
    note: str

@app.get("/")
def read_root():
    return {"message": "Smart Expense AI API is running"}

@app.post("/predict", response_model=ExpenseResponse)
def predict_expense(request: ExpenseRequest):
    try:
        result = analyze_expense(request.text)
        return ExpenseResponse(**result)
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )