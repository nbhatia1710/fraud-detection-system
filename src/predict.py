import pandas as pd
import joblib
import random

# -------- Luhn Algorithm --------
def luhn_check(card_number):
    if not str(card_number).isdigit():
        return False

    digits = [int(d) for d in str(card_number)][::-1]

    total = 0

    for i in range(len(digits)):
        if i % 2 == 1:
            doubled = digits[i] * 2
            if doubled > 9:
                doubled -= 9
            total += doubled
        else:
            total += digits[i]

    return total % 10 == 0


# -------- LOAD MODEL --------
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset (for realistic sampling)
df = pd.read_csv("data/creditcard.csv")
df = df.drop('Time', axis=1)


# -------- PREDICTION FUNCTION --------
def predict_transaction(amount):

    # Smart sampling based on amount
    if amount < 500:
        sample_df = df[df['Class'] == 0]   # mostly safe

    elif amount > 5000:
        sample_df = df[df['Class'] == 1]   # mostly fraud

    else:
        # mix zone
        sample_df = df.sample(frac=1)

    random_row = sample_df.sample(n=1).drop('Class', axis=1)
    transaction = random_row.to_dict(orient='records')[0]

    # Replace amount
    transaction['Amount'] = scaler.transform([[amount]])[0][0]

    input_df = pd.DataFrame([transaction])
    prediction = model.predict(input_df)[0]

    return "Fraudulent Transaction 🚨" if prediction == 1 else "Safe Transaction ✅"

# -------- USER INPUT --------
print("\n💳 Credit Card Fraud Detection System")

while True:
    print("\n-----------------------------")

    card_number = input("Enter Card Number (or type 'exit' to quit): ")

    # Exit condition
    if card_number.lower() == "exit":
        print("\nExiting system... 👋")
        break

    # Luhn validation
    if not luhn_check(card_number):
        print("❌ Invalid Card Number. Try again.")
        continue

    try:
        amount = float(input("Enter Transaction Amount: "))
    except:
        print("❌ Invalid amount. Try again.")
        continue

    # Prediction
    result = predict_transaction(amount)

    print("\nPrediction Result:", result)54
