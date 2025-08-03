from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('lg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map categorical variables
        education_map = {'Graduate': 1, 'Not Graduate': 0}
        self_employed_map = {'Yes': 1, 'No': 0}

        # Get form values
        no_of_dependents = int(request.form['no_of_dependents'])
        education = education_map[request.form['education']]
        self_employed = self_employed_map[request.form['self_employed']]
        income_annum = float(request.form['income_annum'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        cibil_score = float(request.form['cibil_score'])
        residential_assets_value = float(request.form['residential_assets_value'])
        commercial_assets_value = float(request.form['commercial_assets_value'])
        luxury_assets_value = float(request.form['luxury_assets_value'])
        bank_asset_value = float(request.form['bank_asset_value'])

        # Prepare feature array
        input_data = pd.DataFrame([[
            no_of_dependents, education, self_employed, income_annum, loan_amount,
            loan_term, cibil_score, residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value
        ]], columns=[
            'no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
        ])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        result = "Approved" if prediction == 1 else "Rejected"
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
