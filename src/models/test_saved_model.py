import joblib
import pandas as pd

# Load saved model
artifact = joblib.load("models/fraud_detection_logistic.joblib")

model = artifact["model"]
threshold = artifact["threshold"]

# Load test data
df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Select one transaction
sample = X.iloc[[0]]

# Predict fraud probability
probability = model.predict_proba(sample)[0][1]

# Apply saved threshold
prediction = int(probability >= threshold)

print("Fraud Probability:", probability)
print("Threshold:", threshold)
print("Prediction:", prediction)