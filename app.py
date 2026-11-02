from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    carpet_area  = float(data.get('carpet_area', 0))
    bathroom     = float(data.get('bathroom', 0))
    balcony      = float(data.get('balcony', 0))
    car_parking  = float(data.get('car_parking', 0))
    furnishing   = data.get('furnishing', 'Unfurnished')
    transaction  = data.get('transaction', 'Resale')
    ownership    = data.get('ownership', 'Freehold')

    row = {col: 0 for col in columns}
    row['Carpet Area'] = carpet_area
    row['Bathroom']    = bathroom
    row['Balcony']     = balcony
    row['Car Parking'] = car_parking

    if furnishing == 'Semi-Furnished':
        row['Furnishing_Semi-Furnished'] = 1
    elif furnishing == 'Unfurnished':
        row['Furnishing_Unfurnished'] = 1

    if transaction == 'Other':
        row['Transaction_Other'] = 1
    elif transaction == 'Rent/Lease':
        row['Transaction_Rent/Lease'] = 1
    elif transaction == 'Resale':
        row['Transaction_Resale'] = 1

    if ownership == 'Freehold':
        row['Ownership_Freehold'] = 1
    elif ownership == 'Leasehold':
        row['Ownership_Leasehold'] = 1
    elif ownership == 'Power Of Attorney':
        row['Ownership_Power Of Attorney'] = 1

    X = pd.DataFrame([row])[columns]
    prediction = model.predict(X)[0]

    return jsonify({'predicted_price': round(float(prediction), 2)})

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'House Price Prediction API is running'})

if __name__ == '__main__':
    app.run(debug=True)
