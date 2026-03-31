import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Preprocessing
df = df.drop('Time', axis=1)

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SMOTE (only on training)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train model
from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print("===== XGBOOST REPORT =====")
print(classification_report(y_test, y_pred))
# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))