
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# 2. Split data: 80% Train, 10% Validation, 10% Test
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

# 3. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 4. Train on Training data
model.fit(X_train, y_train)

# 5. Predict probabilities on Validation
y_val_proba = model.predict_proba(X_val)[:, 1]

# 6. Business cost assumptions
fraud_loss = 1_000_000
review_cost = 20_000

# 7. Threshold Optimization using Validation only
best_threshold = None
best_cost = float("inf")

for threshold in np.arange(0.01, 1.00, 0.01):

    y_val_pred = (y_val_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_val_pred
    ).ravel()

    total_cost = (
        fn * fraud_loss +
        fp * review_cost
    )

    if total_cost < best_cost:
        best_cost = total_cost
        best_threshold = threshold

print("\n--- Random Forest: Best Threshold ---")
print(f"Threshold: {best_threshold:.2f}")
print(f"Minimum Cost: {best_cost:,.0f} Toman")

# 8. Validation results
y_best_pred = (y_val_proba >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val, y_best_pred
).ravel()

print("\n--- Validation Results ---")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
print(f"Precision: {precision_score(y_val, y_best_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_val, y_best_pred, zero_division=0):.4f}")
print(f"F1-score: {f1_score(y_val, y_best_pred, zero_division=0):.4f}")

# 9. Train final model on Train + Validation
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

final_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_full, y_train_full)

# 10. Final evaluation on untouched Test
y_test_proba = final_model.predict_proba(X_test)[:, 1]

y_test_pred = (y_test_proba >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_test_pred
).ravel()

test_precision = precision_score(
    y_test, y_test_pred, zero_division=0
)

test_recall = recall_score(
    y_test, y_test_pred, zero_division=0
)

test_f1 = f1_score(
    y_test, y_test_pred, zero_division=0
)

test_cost = fn * fraud_loss + fp * review_cost

print("\n--- Random Forest: Final Test Results ---")
print(f"Threshold: {best_threshold:.2f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"Total Cost: {test_cost:,.0f} Toman")

# 11. Save model and threshold
os.makedirs("models", exist_ok=True)

artifact = {
    "model": final_model,
    "threshold": float(best_threshold),
    "model_name": "Random Forest"
}

joblib.dump(
    artifact,
    "models/fraud_detection_random_forest.joblib"
)

print("\nRandom Forest model saved successfully!")
