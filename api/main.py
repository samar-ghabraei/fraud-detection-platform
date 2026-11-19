from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Fraud Detection API",
    description="Credit Card Fraud Detection using Machine Learning",
    version="1.0.0"
)

# Load selected model
artifact = joblib.load(
    "models/fraud_detection_random_forest.joblib"
)

model = artifact["model"]
threshold = artifact["threshold"]


# Input schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame(
        [transaction.model_dump()]
    )

    probability = model.predict_proba(data)[0][1]

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Normal",
        "fraud_probability": round(float(probability), 4),
        "threshold": threshold
    }
