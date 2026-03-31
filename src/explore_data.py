import pandas as pd

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("===== BASIC INFO =====")
print("Shape (rows, columns):", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== CLASS DISTRIBUTION =====")
print(df['Class'].value_counts())

# Optional: percentage distribution
print("\n===== CLASS DISTRIBUTION (%) =====")
print(df['Class'].value_counts(normalize=True) * 100)