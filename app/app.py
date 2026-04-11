import sys
import os

# ✅ ADD SRC FOLDER TO PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

sys.path.append(SRC_PATH)

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_transaction
from luhn import luhn_check

server = Flask(__name__)
CORS(server)


@server.route('/')
def home():
    return jsonify({"status": "API is running 🚀"})


@server.route('/validate', methods=['POST'])
def validate():
    data = request.get_json()
    card = data.get('card_number', '').replace(' ', '')
    return jsonify({'valid': luhn_check(card)})


@server.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    amount = float(data.get('amount', 0))
    result = predict_transaction(amount)
    return jsonify(result)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    server.run(host='0.0.0.0', port=port)