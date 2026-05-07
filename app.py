from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import json
import os
from src.utils.main_utils import MainUtils
from src.pipeline.predict_pipeline import PredictPipeline


app = Flask(__name__)

# Load top features and means
with open("artifacts/top_features.json") as f:
    feature_info = json.load(f)
top_features = feature_info["top_features"]
feature_means = feature_info["feature_means"]

# Load preprocessor and model
preprocessor = MainUtils().load_object("artifacts/preprocessor.pkl")
model = MainUtils().load_object("artifacts/model.pkl")
all_features = preprocessor['numeric_cols'] + preprocessor['categorical_columns']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # 1. Retrieve the user inputs
            salary = float(request.form.get('salary', 0))
            loan_amount = float(request.form.get('loan_amount', 0))
            credit_score = float(request.form.get('credit_score', 0))
            years_of_experience = float(request.form.get('years_of_experience', 0))
            age = float(request.form.get('age', 0))
            job_status = request.form.get('job_status', '')
            interest_rate = float(request.form.get('interest_rate', 10.0))
            tenure = float(request.form.get('tenure', 5))

            # Calculate EMI and Total Amount
            if loan_amount > 0 and interest_rate > 0 and tenure > 0:
                P = loan_amount
                R = interest_rate / (12 * 100)
                N = tenure * 12
                emi = P * R * ((1 + R)**N) / (((1 + R)**N) - 1)
                total_amount = emi * N
            else:
                emi = 0
                total_amount = loan_amount
                N = tenure * 12 if tenure > 0 else 60

            # 2. Custom AI Prediction Logic
            reasons = []
            
            # Check Debt-to-Income Ratio (EMI should not exceed 50% of monthly salary)
            monthly_salary = salary / 12
            if monthly_salary > 0:
                dti = emi / monthly_salary
                if dti > 0.5:
                    reasons.append(f"Your EMI (₹{int(emi):,}) exceeds 50% of your estimated monthly salary (₹{int(monthly_salary):,}).")
            else:
                reasons.append("Invalid salary amount.")
                
            # Check Credit Score
            if credit_score < 600:
                reasons.append(f"Your credit score ({int(credit_score)}) is below our minimum requirement of 600.")
                
            # Check Age
            if age < 18:
                reasons.append("You must be at least 18 years old to apply for a loan.")
            elif age + tenure > 65:
                reasons.append("Your age at the end of the loan tenure exceeds our maximum limit of 65 years.")
                
            # Check Job Status / Stability
            if job_status in ['Unemployed', 'Student']:
                reasons.append(f"We currently do not offer loans to individuals with '{job_status}' status.")
            elif years_of_experience < 1 and job_status != 'Pensioner':
                reasons.append("You need at least 1 year of work experience to qualify.")

            if len(reasons) == 0:
                status = "Loan Approved"
            else:
                status = "Loan Rejected"

            return render_template('result.html', status=status, reasons=reasons, emi=round(emi, 2), tenure=int(N), loan_amount=loan_amount, total_amount=round(total_amount, 2))
            
        except Exception as e:
            # If there's an error, fall back to the form with an error message
            return render_template('form.html', error=str(e))
            
    return render_template('form.html')


@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    output_path = None
    if request.method == 'POST':
        file = request.files['file']
        upload_dir = os.path.join("artifacts", "prediction_artifacts")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        pipeline = PredictPipeline()
        output_path = pipeline.predict_from_csv(file_path)
    return render_template('upload.html', output_path=output_path)

@app.route('/download/<filename>')
def download_file(filename):
    predictions_dir = os.path.join("artifacts", "predictions")
    return send_from_directory(predictions_dir, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)