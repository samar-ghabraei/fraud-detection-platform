import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# 1. Load dataset
df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# 2. Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# 3. Create model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# 4. Train model
model.fit(X_train, y_train)

# 5. Predict probabilities and classes
#y_proba = model.predict_proba(X_test)[:, 1]
#y_pred = model.predict(X_test)
y_val_proba = model.predict_proba(X_val)[:, 1]


# 6. Evaluate model
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nPrecision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1-score:", f1_score(y_test, y_pred))
# print("PR-AUC (Average Precision):", average_precision_score(y_test, y_proba))

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_val_pred = (y_val_proba >= threshold ).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()

    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)

    print(f"\nThreshold: {threshold}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# # 7. Business cost analysis

# fraud_loss = 1_000_000
# review_cost = 20_000

# print("\n--- Business Cost Analysis ---")

# for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:

#     y_val_pred = (y_val_proba >= threshold).astype(int)

#     tn, fp, fn, tp = confusion_matrix(
#         y_val, y_val_pred
#     ).ravel()

#     total_cost = (
#         fn * fraud_loss +
#         fp * review_cost
#     )

#     print(f"\nThreshold: {threshold}")
#     print(f"Missed fraud (FN): {fn}")
#     print(f"False alarms (FP): {fp}")
#     print(f"Total cost: {total_cost:,} Toman")

# 7. Threshold Optimization

# 7. Threshold Optimization

fraud_loss = 1_000_000
review_cost = 20_000

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

print("\n--- Best Threshold ---")
print(f"Threshold: {best_threshold:.2f}")
print(f"Minimum Cost: {best_cost:,.0f} Toman")

# Evaluate the best threshold
y_best_pred = (y_val_proba >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val, y_best_pred
).ravel()

print("\n--- Best Threshold Results ---")
print(f"Threshold: {best_threshold:.2f}")
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TN: {tn}")
print(f"Precision: {precision_score(y_val, y_best_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_val, y_best_pred, zero_division=0):.4f}")
print(f"Minimum Cost: {best_cost:,.0f} Toman")


# 8. Final Model Evaluation on Test
# Combine training and validation data
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

# Train final model
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

final_model.fit(X_train_full, y_train_full)

# Predict on untouched test data
y_test_proba = final_model.predict_proba(X_test)[:, 1]

# Use threshold selected on validation
y_test_pred = (y_test_proba >= best_threshold).astype(int)

# Calculate final metrics
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

test_cost = (
    fn * fraud_loss +
    fp * review_cost
)

print("\n--- Final Test Results ---")
print(f"Threshold: {best_threshold:.2f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"Total Cost: {test_cost:,.0f} Toman")

