# Credit Risk AI - Intelligent Loan Assessment System

A modern, AI-powered credit risk assessment system that predicts loan default probability and provides explainable AI insights using a trained LightGBM model.

## 🚀 Features

- **AI-Powered Risk Assessment**: Uses a trained LightGBM model to predict loan default probability
- **Explainable AI**: Provides SHAP-based feature explanations showing why decisions are made
- **Risk-Based Pricing**: Automatically assigns interest rates based on default probability
- **Profit Analysis**: Calculates expected profit for each loan application
- **Modern UI**: Clean, responsive web interface with intuitive form inputs
- **Real-time Predictions**: Instant results with detailed breakdowns

## 🏗️ Architecture

The system consists of:

1. **Frontend**: Modern HTML5/CSS3/JavaScript interface
2. **Backend**: Flask API server with LightGBM model integration
3. **ML Model**: Pre-trained LightGBM classifier with engineered features
4. **Explainability**: SHAP values for feature importance and decision transparency

## 📋 Prerequisites

- Python 3.8+
- pip package manager
- Modern web browser

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**:
   - `lightgbm_credit_risk_model.pkl` - Trained LightGBM model
   - `label_encoders.pkl` - Categorical feature encoders
   - `feature_scaler.pkl` - Feature scaling parameters

## 🚀 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:7860
   ```

3. **Fill out the application form** with applicant information:
   - Personal details (age, income, employment)
   - Loan information (amount, purpose, grade)
   - Credit history details

4. **Click "Assess Credit Risk"** to get instant predictions

## 📊 Model Features

The system uses the following features for prediction:

### Personal Information
- **Age**: Applicant's age in years
- **Annual Income**: Total yearly income in dollars
- **Home Ownership**: RENT, OWN, MORTGAGE, or OTHER
- **Employment Length**: Years of employment experience

### Loan Details
- **Loan Amount**: Requested loan amount in dollars
- **Loan Intent**: Purpose of the loan (Personal, Education, Medical, etc.)
- **Loan Grade**: Credit grade from A (Excellent) to G (Worst)
- **Debt-to-Income Ratio**: Monthly debt payments as percentage of income

### Credit History
- **Previous Default**: Whether applicant has defaulted before (Y/N)
- **Credit History Length**: Years of credit history

### Engineered Features
- **Liquidity Ratio**: Income relative to loan amount
- **Debt Burden**: Income × debt-to-income ratio
- **Experience Ratio**: Employment length relative to age
- **Income Stability**: Employment length × income
- **Credit Density**: Credit history length relative to age
- **Risk Capacity**: Available income after debt payments
- **DTI Squared**: Non-linear debt-to-income relationship
- **Income-to-Loan Ratio**: Income relative to loan amount

## 🎯 Output Explanation

### Prediction Results
- **Default Status**: Binary prediction (Default/Non-Default)
- **Default Probability**: Percentage chance of loan default (0-100%)
- **Interest Rate**: Risk-based interest rate (7-18%)
- **Expected Profit**: Calculated profit/loss for the lender

### Feature Explanations
The system provides SHAP-based explanations showing:
- **Top Contributing Features**: Most important factors in the decision
- **Feature Values**: Actual values for each feature
- **Contribution**: How much each feature influenced the prediction

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: LightGBM (Gradient Boosting Decision Trees)
- **Training**: Optimized hyperparameters using Optuna
- **Performance**: High accuracy with ROC-AUC optimization
- **Explainability**: SHAP values for model interpretability

### API Endpoints
- `GET /` - Main application interface
- `POST /api/predict` - Credit risk prediction
- `GET /api/features` - Available features and types

### Data Preprocessing
- **Feature Engineering**: Creates 8 additional engineered features
- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Scaling**: Standardization for numeric features
- **Missing Value Handling**: Robust preprocessing pipeline

## 📈 Business Logic

### Interest Rate Assignment
The system uses a tiered approach:
- **Low Risk (PD ≤ 10%)**: 7% + (15 × PD)
- **Medium Risk (10% < PD ≤ 40%)**: 8.5% + (20 × (PD - 0.10))
- **High Risk (PD > 40%)**: 14.5% + (6 × (PD - 0.40))

### Expected Profit Calculation
```
Expected Profit = (1 - PD) × (Interest Rate × Loan Amount) - (PD × LGD × Loan Amount)
```
Where LGD (Loss Given Default) = 60%

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile devices
- **Modern Styling**: Clean, professional appearance with gradients
- **Interactive Elements**: Hover effects, loading states, and animations
- **Form Validation**: Client-side validation with helpful error messages
- **Real-time Updates**: Instant results display with smooth transitions

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Error**:
   - Ensure all `.pkl` files are in the project directory
   - Check file permissions and paths

2. **Package Installation Issues**:
   - Use virtual environment: `python -m venv venv`
   - Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)

3. **Port Already in Use**:
   - Change port in `app.py`: `app.run(port=7861)`
   - Or kill existing process using the port

### Performance Tips
- The system loads the model once at startup
- SHAP explanations may take a few seconds for complex cases
- Use appropriate input ranges for best results

## 🤝 Contributing

To improve the system:
1. Enhance the UI/UX design
2. Add more sophisticated feature engineering
3. Implement additional ML models for comparison
4. Add data visualization and analytics
5. Improve error handling and validation

## 📄 License

This project is for educational and demonstration purposes. The trained model and data processing pipeline are based on the Credit Risk Dataset.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments and documentation
3. Ensure all dependencies are properly installed
4. Verify model files are present and accessible

---

**Note**: This system is designed for educational and demonstration purposes. For production use, ensure proper security measures, data validation, and compliance with relevant regulations.
