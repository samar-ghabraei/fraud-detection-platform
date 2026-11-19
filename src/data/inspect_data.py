import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print("Dataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nFraud percentage:")
print(df["Class"].mean() * 100)

print("\nFraud rows among duplicates:")
print(df[df.duplicated()]["Class"].value_counts())

print("\nAmount statistics: ")
print(df["Amount"].describe())

print("\nAverage amount by class: ")
print(df.groupby("Class")["Amount"].mean())

print("\nFraud vs Normal percentage:")
print(df["Class"].value_counts(normalize=True)*100)
