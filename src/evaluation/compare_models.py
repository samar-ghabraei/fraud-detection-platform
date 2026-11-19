import pandas as pd
import joblib
import os

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)   

# 1. Load dataset
df = pd.read_csv("data/creditcard.csv") 

X = df.drop(columns=["Class"])
y = df["Class"]

# 2. Receate the same Test split
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# 3. Load saved models
model_files = {
    "Logistic Regression": "models/fraud_detection_logistic.joblib",
    "Random Forest": "models/fraud_detection_random_forest.joblib",
    "XGBoost": "models/fraud_detection_xgboost.joblib"
}

results = []

# 4. Evaluate each model
for model_name, file_path in model_files.items():
    aertifact = joblib.load(file_path)

    model = aertifact["model"]
    threshold = aertifact["threshold"]  

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fraud_loss = 1_000_000
    review_cost = 20_000

    total_cost = (
        fn * fraud_loss +
        fp * review_cost
    )

    results.append({
        "Model": model_name,
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Total Cost (Toman)": total_cost
    })

# 5. Create comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Total Cost (Toman)"
)

print("\n--- Model Comparison ---")
print(results_df.to_string(index=False))

# 6. Save results
os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nComparison report saved successfully!")
