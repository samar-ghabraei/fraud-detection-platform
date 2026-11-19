import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining fraud cases:")
print(y_train.value_counts())

print("\nTesting fraud cases:")
print(y_test.value_counts())