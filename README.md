# Hackathon
#  Credit Risk Scorecard

## Overview
This project tackles a critical challenge in the financial industry: accurately assessing the risk of loan default to optimize lending decisions. By leveraging machine learning, we developed a system that predicts an applicant's Probability of Default (PD), assigns risk-based interest rates, and identifies the most profitable portfolio of loans, thereby maximizing returns while managing risk for financial institutions.

## Business Problem
Traditional rule-based credit scoring methods are often inadequate, leading to either significant financial losses from bad loans or missed opportunities from rejecting good customers. Our solution uses advanced ML to analyze historical data, predict default risk with high accuracy, and provide data-driven recommendations for approval, pricing, and portfolio construction.

## Solution Architecture
The project follows a structured, end-to-end pipeline:
1.  **Data Acquisition & Analysis:** Load and explore the loan application dataset.
2.  **Feature Engineering:** Create powerful new features that capture complex financial relationships.
3.  **Data Preprocessing:** Encode categorical variables and scale numerical features.
4.  **Model Building & Tuning:** Train a LightGBM model and optimize its hyperparameters using Optuna for maximum predictive power (ROC-AUC: 0.951).
5.  **Inference & Business Application:** Generate PD scores for all applicants, assign interest rates (7%-18%), calculate expected profit, and select the top 30% most profitable applicants.
6.  **Deployment Preparation:** Save the final model and all preprocessing objects for production use.



## Technologies & Libraries
-   **Language:** Python 3.8+
-   **Data Manipulation:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn, LightGBM
-   **Hyperparameter Tuning:** Optuna
-   **Visualization:** Matplotlib, Seaborn
-   **Serialization:** Joblib

## Installation & Execution (Google Colab)

1. **Open the Notebook in Colab:**  
   - Upload [`hackathon.ipynb`][def] to Google Colab **or**  
   - Click the Colab badge to open directly  


2. **Install Dependencies (inside Colab):**  
   Run the following command in the first cell to install all required libraries:  
   ```python
   %pip install pandas numpy scikit-learn lightgbm optuna joblib
   or
   pip install -r requirements.txt
## Key Results
-   **Model Performance:** Achieved an exceptional ROC-AUC score of **0.951**.
-   **Business Impact:** Identified a portfolio with an **expected profit of $7.87 million** from the top 30% of applicants.
-   **Risk-Based Pricing:** Assigned competitive interest rates with an **average of 7.34%** for the selected low-risk portfolio.

# Interest Rate Strategy & Portfolio Optimization

## Question 2: Interest Rate Assignment Strategy
The assignment of risk-based interest rates is critical for both **profitability** and **competitiveness**.  
We designed a **three-tiered pricing algorithm** to achieve:

- **Competitiveness**: Attractive rates for top applicants.  
- **Risk-Based Pricing**: Ensure profitability across the spectrum.  
- **Behavioral Incentives**: Offer higher (yet feasible) rates to higher-risk applicants.  

### Tiered Interest Rate Function

**Interest Rate** = 
- f(PD) = [7 + 15PD], for PD ∈ [0, 0.10]
- f(PD) = [8.5 + 20(PD-0.10)], for PD ∈ (0.10, 0.40]
- f(PD) = [14.5 + 6*(PD-0.40)], for PD ∈ (0.40, 1.0]

Capped between 7% (floor) and 18% (ceiling).


### Tier Rationale
- **Tier 1 (Super-Prime, PD ≤ 10%)** → Minimal premium, highly competitive.  
- **Tier 2 (Prime, 10% < PD ≤ 40%)** → Steep slope for strong risk-reward balance.  
- **Tier 3 (Sub-Prime, PD > 40%)** → Flatter slope, effectively soft decline via pricing.  

This **non-linear, tiered model** balances acquisition with **risk-adjusted profitability**.

---

## Question 3: Maximum Profit & Portfolio Optimization
We shift focus from **risk minimization** to **profit maximization** by selecting applicants that yield the **highest expected profit**, not just lowest PD.  

### Expected Profit Function
**E[Profit] = (Profit_Non_Default * (1 - PD)) + (Loss_Default * PD)**

Where:
- Profit_Non_Default = Loan_Amount * (Assigned_Rate / 100)
- Loss_Default = -Loan_Amount * LGD (assuming LGD = 60%)

### Optimization Steps
1. **Generate PDs** → Predict with model.  
2. **Assign Rates** → Using the tiered strategy.  
3. **Compute E[Profit]** → For all applicants.  
4. **Rank Applicants** → By profitability (highest → lowest).  
5. **Apply Constraint** → Select **top 30%** applicants only.  
6. **Sum Profits** → Compute total portfolio value.  

### Results
- **Accepted Applicants**: 9,774 (Top 30%)  
- **Total Expected Profit**: **$7,873,174.08**  
- **Avg. PD (Portfolio)**: 2.28%  
- **Avg. Interest Rate**: 7.34%  

---

## 🚀 Key Takeaway
Our approach leverages **tiered risk-based pricing** + **profit-driven portfolio selection** to maximize both **competitiveness** and **financial returns**.



[def]: https://colab.research.google.com/drive/1Wn0Ox3JQjyFwgxXAZj1t5MZmlKv_QXpS?usp=sharing
