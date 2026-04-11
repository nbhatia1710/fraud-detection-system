import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset (IMPORTANT: you must have this file)
df = pd.read_csv("data/creditcard.csv")

# Preprocess
df = df.drop('Time', axis=1)

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

X = df.drop('Class', axis=1)
y = df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train_res, y_train_res)

# Save
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model + Scaler saved in models/")