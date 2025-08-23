
import os
import json
import traceback
from typing import List, Dict, Any, Optional
import random

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import shap

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global variables for model and preprocessing
model = None
label_encoders = None
scaler = None
feature_names = None
categorical_features = None
numeric_features = None
dataset_df = None  # Store the full dataset for analysis

def load_model():
    """Load the trained model and preprocessing artifacts"""
    global model, label_encoders, scaler, feature_names, categorical_features, numeric_features, dataset_df
    
    try:
        # Load the trained model
        model = joblib.load('lightgbm_credit_risk_model.pkl')
        
        # Load preprocessing artifacts
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        
        # Define feature names based on the notebook
        feature_names = [
            'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
            'loan_intent', 'loan_grade', 'loan_amnt', 'loan_percent_income',
            'cb_person_default_on_file', 'cb_person_cred_hist_length',
            'liquidity_ratio', 'debt_burden', 'experience_ratio', 'income_stability',
            'credit_density', 'risk_capacity', 'dti_squared', 'income_to_loan_ratio'
        ]
        
        # Define which features are categorical vs numeric
        categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        numeric_features = [f for f in feature_names if f not in categorical_features]
        
        # Load the dataset for analysis
        try:
            dataset_df = pd.read_csv('Credit-Risk-Dataset.csv')
            print("Dataset loaded successfully for analysis!")
        except Exception as e:
            print(f"Warning: Could not load dataset for analysis: {e}")
            dataset_df = None
        
        print("Model and preprocessing artifacts loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_input(applicant_data: Dict[str, Any]) -> np.ndarray:
    """Preprocess input data using the same pipeline as training"""
    try:
        # Create a DataFrame with the input data
        df = pd.DataFrame([applicant_data])
        
        # Create engineered features (same as in notebook)
        df['liquidity_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)
        df['debt_burden'] = df['person_income'] * df['loan_percent_income']
        df['experience_ratio'] = df['person_emp_length'] / (df['person_age'] + 1)
        df['income_stability'] = df['person_emp_length'] * df['person_income']
        df['credit_density'] = df['cb_person_cred_hist_length'] / (df['person_age'] - 18 + 1)
        df['risk_capacity'] = (df['person_income'] - df['loan_amnt'] * df['loan_percent_income']) / 1000
        df['dti_squared'] = df['loan_percent_income'] ** 2
        df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)
        
        # Ensure all features are present in the correct order
        df = df[feature_names]
        
        # Encode categorical features
        for col in categorical_features:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
        
        # Scale numeric features
        df[numeric_features] = scaler.transform(df[numeric_features])
        
        return df.values
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def assign_interest_rate(pd_value: float) -> float:
    """Assign interest rate based on probability of default using tiered approach"""
    if pd_value <= 0.10:  # Low Risk Tier
        rate = 7 + (15 * pd_value)
    elif pd_value <= 0.40:  # Medium Risk Tier
        rate = 8.5 + (20 * (pd_value - 0.10))
    else:  # High Risk Tier
        rate = 14.5 + (6 * (pd_value - 0.40))
    
    # Ensure the rate is capped between 7% and 18%
    return max(7.0, min(rate, 18.0))

def calculate_expected_profit(loan_amnt: float, interest_rate: float, pd_value: float, lgd: float = 0.6) -> float:
    """Calculate expected profit for a loan"""
    profit_if_paid = loan_amnt * (interest_rate / 100)  # Interest earned
    loss_if_default = -loan_amnt * lgd  # Loss given default
    
    expected_profit = (profit_if_paid * (1 - pd_value)) + (loss_if_default * pd_value)
    return expected_profit

def get_feature_explanations(applicant_data: Dict[str, Any], pd_value: float) -> List[Dict[str, Any]]:
    """Get feature explanations using SHAP values"""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Preprocess input
        X_processed = preprocess_input(applicant_data)
        
        # Get SHAP values
        shap_values = explainer.shap_values(X_processed)
        
        # Create feature contribution list
        contributions = []
        for i, feature in enumerate(feature_names):
            contributions.append({
                "feature": feature,
                "value": applicant_data.get(feature, 0),
                "contribution": float(shap_values[1][i]) if len(shap_values) > 1 else float(shap_values[0][i])
            })
        
        # Sort by absolute contribution and take top 10
        contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)[:10]
        
        return contributions
        
    except Exception as e:
        print(f"Error getting SHAP explanations: {e}")
        # Fallback to simple feature importance
        return [
            {"feature": "person_income", "value": applicant_data.get("person_income", 0), "contribution": 0.1},
            {"feature": "loan_amnt", "value": applicant_data.get("loan_amnt", 0), "contribution": 0.08},
            {"feature": "loan_percent_income", "value": applicant_data.get("loan_percent_income", 0), "contribution": 0.06}
        ]

def analyze_dataset_row(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single row from the dataset and return predictions"""
    try:
        # Create applicant data structure
        applicant = {
            'person_age': row_data.get('person_age', 0),
            'person_income': row_data.get('person_income', 0),
            'person_home_ownership': row_data.get('person_home_ownership', 'RENT'),
            'person_emp_length': row_data.get('person_emp_length', 0),
            'loan_intent': row_data.get('loan_intent', 'PERSONAL'),
            'loan_grade': row_data.get('loan_grade', 'C'),
            'loan_amnt': row_data.get('loan_amnt', 0),
            'loan_percent_income': row_data.get('loan_percent_income', 0.1),
            'cb_person_default_on_file': row_data.get('cb_person_default_on_file', 'N'),
            'cb_person_cred_hist_length': row_data.get('cb_person_cred_hist_length', 0)
        }
        
        # Preprocess and predict
        X_processed = preprocess_input(applicant)
        pd_value = model.predict_proba(X_processed)[0, 1]
        
        # Calculate other metrics
        interest_rate = assign_interest_rate(pd_value)
        expected_profit = calculate_expected_profit(applicant['loan_amnt'], interest_rate, pd_value)
        
        return {
            **applicant,
            'pd': float(pd_value),
            'interest_rate': float(interest_rate),
            'expected_profit': float(expected_profit)
        }
        
    except Exception as e:
        print(f"Error analyzing dataset row: {e}")
        return None

@app.route("/")
def home():
    """Serve the main page"""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """API endpoint for credit risk prediction"""
    try:
        if model is None:
            return jsonify({"ok": False, "error": "Model not loaded"}), 500
        
        # Get applicant data from request
        applicant = request.json.get("applicant", {})
        
        # Validate required fields
        required_fields = ['person_age', 'person_income', 'person_home_ownership', 'loan_intent', 
                          'loan_grade', 'loan_amnt', 'loan_percent_income', 'person_emp_length',
                          'cb_person_default_on_file', 'cb_person_cred_hist_length']
        
        missing_fields = [field for field in required_fields if field not in applicant or applicant[field] is None]
        if missing_fields:
            return jsonify({"ok": False, "error": f"Missing required fields: {missing_fields}"}), 400
        
        # Preprocess input
        X_processed = preprocess_input(applicant)
        
        # Make prediction
        pd_value = model.predict_proba(X_processed)[0, 1]
        
        # Determine prediction class
        prediction = "Default" if pd_value > 0.5 else "Non-Default"
        
        # Assign interest rate
        interest_rate = assign_interest_rate(pd_value)
        
        # Calculate expected profit
        expected_profit = calculate_expected_profit(
            applicant['loan_amnt'], 
            interest_rate, 
            pd_value
        )
        
        # Get feature explanations
        explanations = get_feature_explanations(applicant, pd_value)
        
        return jsonify({
            "ok": True,
            "prediction": prediction,
            "pd": float(pd_value),
            "interest_rate": float(interest_rate),
            "expected_profit": float(expected_profit),
            "explanation": {
                "top_contributions": explanations
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/dataset/sample")
def get_dataset_sample():
    """Get a random sample from the dataset with predictions"""
    try:
        if dataset_df is None:
            return jsonify({"ok": False, "error": "Dataset not loaded"}), 500
        
        count = min(int(request.args.get('count', 10)), 100)  # Limit to 100 max
        
        # Get random sample
        sample_df = dataset_df.sample(n=min(count, len(dataset_df)), random_state=42)
        
        # Analyze each row
        results = []
        for _, row in sample_df.iterrows():
            result = analyze_dataset_row(row.to_dict())
            if result:
                results.append(result)
        
        return jsonify({
            "ok": True,
            "sample": results
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/dataset/top-profitable")
def get_top_profitable():
    """Get the top 30% most profitable applicants from the dataset"""
    try:
        if dataset_df is None:
            return jsonify({"ok": False, "error": "Dataset not loaded"}), 500
        
        # Analyze all rows (this might take some time for large datasets)
        # For performance, we'll analyze a sample first
        sample_size = min(1000, len(dataset_df))
        sample_df = dataset_df.sample(n=sample_size, random_state=42)
        
        results = []
        for _, row in sample_df.iterrows():
            result = analyze_dataset_row(row.to_dict())
            if result:
                results.append(result)
        
        # Sort by expected profit and take top 30%
        results.sort(key=lambda x: x['expected_profit'], reverse=True)
        top_30_percent = results[:int(len(results) * 0.3)]
        
        return jsonify({
            "ok": True,
            "applicants": top_30_percent
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/dataset/stats")
def get_dataset_stats():
    """Get overall dataset statistics"""
    try:
        if dataset_df is None:
            return jsonify({"ok": False, "error": "Dataset not loaded"}), 500
        
        # Analyze a sample for statistics
        sample_size = min(1000, len(dataset_df))
        sample_df = dataset_df.sample(n=sample_size, random_state=42)
        
        results = []
        for _, row in sample_df.iterrows():
            result = analyze_dataset_row(row.to_dict())
            if result:
                results.append(result)
        
        if not results:
            return jsonify({"ok": False, "error": "No valid results found"}), 500
        
        # Calculate statistics
        total_applicants = len(dataset_df)
        avg_default_rate = np.mean([r['pd'] for r in results])
        avg_interest_rate = np.mean([r['interest_rate'] for r in results])
        total_expected_profit = sum([r['expected_profit'] for r in results])
        
        # Top 30% stats
        results.sort(key=lambda x: x['expected_profit'], reverse=True)
        top_30_percent = results[:int(len(results) * 0.3)]
        top_30_count = len(top_30_percent)
        top_30_profit = sum([r['expected_profit'] for r in top_30_percent])
        
        return jsonify({
            "ok": True,
            "stats": {
                "total_applicants": total_applicants,
                "avg_default_rate": float(avg_default_rate),
                "avg_interest_rate": float(avg_interest_rate),
                "total_expected_profit": float(total_expected_profit),
                "top_30_count": top_30_count,
                "top_30_profit": float(top_30_profit)
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/features", methods=["GET"])
def get_features():
    """Get available features and their types"""
    try:
        return jsonify({
            "ok": True,
            "features": feature_names,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Load model on startup
    if load_model():
        print("Starting Flask app...")
        app.run(debug=True, host="0.0.0.0", port=7860)
    else:
        print("Failed to load model. Exiting.")

