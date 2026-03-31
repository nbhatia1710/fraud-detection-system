import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Original Shape:", df.shape)

# 1. Drop 'Time' column
df = df.drop('Time', axis=1)

# 2. Scale 'Amount'
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# 3. Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from imblearn.over_sampling import SMOTE

# Apply SMOTE ONLY on training data
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("Training shape:", X_train_res.shape)
print("Class distribution after SMOTE:")
print(y_train_res.value_counts())

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)