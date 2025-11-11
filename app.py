import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from dataclasses import dataclass
import warnings
import mlflow.sklearn  # <--- NEW: MLflow Import Added
warnings.filterwarnings('ignore')

# --- ANIMATED RGB BACKGROUND FUNCTION (MORE INTERESTING) ---
def set_animated_background():
    """
    Sets a smoothly transitioning, animated gradient background 
    with a wider color palette and a more dynamic animation speed.
    """
    st.markdown(
        """
        <style>
        @keyframes gradient-animation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        
        .stApp {
            /* 1. ANIMATION: Gradient with a broader, light spectrum */
            /* Colors used: Light Red, Light Orange, Light Yellow, Light Green, Light Blue */
            background: linear-gradient(
                270deg, 
                #ffb7b7, /* Pastel Red */
                #ffc5a1, /* Pastel Orange */
                #fffac1, /* Pastel Yellow */
                #c1ffc9, /* Pastel Green */
                #b7f0ff  /* Pastel Blue */
            );
            background-size: 800% 800%; /* Wider size for smoother, sweeping movement */
            animation: gradient-animation 15s ease infinite; /* Faster 15s cycle time */
            
            /* 2. VISIBILITY FIX: Set main text color to dark gray/black */
            color: #333333; 
        }
        
        /* Ensure all standard text within Streamlit containers is dark */
        h1, h2, h3, h4, p, label, .stMarkdown, .stText, .stForm {
            color: #333333 !important;
        }
        
        /* FIX FOR BUTTON TEXT VISIBILITY */
        .stButton > button {
            color: #333333 !important; /* Forces dark gray for button text */
        }

        /* Keep the sidebar static and also set dark text */
        .stSidebar {
            background-color: #f7f7f7; 
            background-attachment: fixed;
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# ----------------------------------------------------------------
# ----------------------------------------


# --- 1. CONFIGURATION AND UTILITIES (Copied from Prediction Pipeline) ---

@st.cache_resource
def load_object(file_path):
    """Loads a Python object from a binary file."""
    try:
        # NOTE: This function is now ONLY used for the preprocessor.pkl
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        st.error(f"Error loading artifact: {file_path}. Please ensure all .pkl files are in the 'artifacts' folder.")
        st.stop()
        
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all the custom feature engineering logic."""
    df_copy = df.copy()
    
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                    'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount']
    
    for col in expense_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
    
    df_copy['total_monthly_expenses'] = df_copy[expense_cols].sum(axis=1)
    
    safe_salary = pd.to_numeric(df_copy['monthly_salary'], errors='coerce').replace(0, 1)

    df_copy['DTI_ratio'] = df_copy['current_emi_amount'] / safe_salary
    df_copy['ETI_ratio'] = df_copy['total_monthly_expenses'] / safe_salary
    df_copy['free_cash_flow'] = df_copy['monthly_salary'] - df_copy['total_monthly_expenses']

    df_copy['income_per_capita'] = df_copy['monthly_salary'] / df_copy['family_size'].replace(0, 1)
    df_copy['income_per_dependent'] = df_copy['monthly_salary'] / (df_copy['dependents'].replace(0, 0) + 1)
    
    df_copy['job_stability_ratio'] = df_copy['years_of_employment'] / df_copy['age'].replace(0, 1)
    
    return df_copy

class CustomData:
    """Class to gather user input from the Streamlit form."""
    def __init__(self, **kwargs):
        self.data = kwargs
        
    def get_data_as_dataframe(self):
        df = pd.DataFrame([self.data])
        for col in ['monthly_salary', 'bank_balance', 'emergency_fund', 'requested_amount', 'current_emi_amount', 'credit_score', 'age', 'years_of_employment', 'family_size', 'dependents', 'requested_tenure']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        return df


class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")
        
        # --- MLflow URI using the confirmed RUN ID and artifact path 'model' ---
        self.mlflow_model_uri = "mlruns/0/978349954066487471/18a988f2a822498098b2545646b54ffa/models/m-4e63d19aaeb34b593b6502d/artifacts" 
        # ----------------------------------------------------------------------

    @st.cache_resource
    def load_artifacts(_self):
        """Loads the preprocessor (pickle) and the trained regression model (MLflow)."""
        
        # Load preprocessor using standard pickle function
        preprocessor = load_object(_self.preprocessor_path)
        
        # Load the model using MLflow's standard API
        try:
            reg_model = mlflow.sklearn.load_model(_self.mlflow_model_uri)
        except Exception as e:
            st.error(f"Error loading MLflow model from URI: {_self.mlflow_model_uri}. Ensure 'mlruns' directory is in the root and the RUN ID is correct.")
            st.stop()
            
        return preprocessor, reg_model

    def predict_loan_decision(self, applicant_data: pd.DataFrame, requested_emi_amount: float) -> dict:
        
        preprocessor, reg_model = self.load_artifacts()

        # 1. Feature Engineering
        processed_df = add_engineered_features(applicant_data)
        
        # 2. Transformation
        data_scaled = preprocessor.transform(processed_df)

        # 3. Predict Max Monthly EMI (Regression Target)
        max_monthly_emi = reg_model.predict(data_scaled)[0]
        max_monthly_emi = max_monthly_emi if max_monthly_emi > 0 else 0 

        # 4. Determine Final Decision based on Regression Output
        final_decision = "Ineligible (Max EMI < Requested EMI)"
        loan_eligible = 0
        
        if max_monthly_emi >= requested_emi_amount:
            final_decision = "Eligible"
            loan_eligible = 1
        
        return {
            "Loan Eligibility": "ELIGIBLE" if loan_eligible == 1 else "INELIGIBLE",
            "Predicted Max Monthly EMI": round(max_monthly_emi, 2),
            "Requested EMI": requested_emi_amount,
            "Decision Reason": final_decision
        }


# --- 2. STREAMLIT FRONTEND LOGIC ---

def main():
    # --- CALL THE ANIMATED BACKGROUND FUNCTION HERE ---
    set_animated_background() 
    # --------------------------------------------------
    
    st.set_page_config(page_title="EMI Predict AI", layout="wide")
    st.title("💰 EMI Predict AI: Loan Eligibility Predictor")
    st.subheader("Powered by Regression Model (R²: 0.9708) - MLflow Managed") # Updated R^2 from last run

    
    # Initialize the pipeline
    pipeline = PredictionPipeline()

    with st.form(key='loan_form'):
        st.header("Applicant Financial Profile")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Core Details")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ('Male', 'Female'))
            marital_status = st.selectbox("Marital Status", ('Married', 'Single'))
            education = st.selectbox("Education", ('Graduate', 'Post-Graduate', 'Under-Graduate', 'No Formal Education'))
            requested_tenure = st.slider("Requested Tenure (Months)", min_value=12, max_value=120, value=60)
        
        with col2:
            st.markdown("### Income & Debt")
            monthly_salary = st.number_input("Monthly Salary (₹)", min_value=1000.0, value=75000.0, step=1000.0)
            credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=780)
            years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=50, value=8)
            existing_loans = st.number_input("Existing Loans Count", min_value=0, value=1)
            current_emi_amount = st.number_input("Current Total EMI Amount (₹)", min_value=0.0, value=15000.0, step=100.0)

        with col3:
            st.markdown("### Family & Home")
            family_size = st.number_input("Family Size", min_value=1, value=4)
            dependents = st.number_input("Dependents", min_value=0, value=2)
            house_type = st.selectbox("House Type", ('Rented', 'Owned', 'Other'))
            monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0.0, value=10000.0, step=100.0)
            employment_type = st.selectbox("Employment Type", ('Salaried', 'Self-Employed', 'Business'))
            company_type = st.selectbox("Company Type", ('Service', 'Manufacturing', 'Other'))
        
        st.markdown("---")
        st.header("Loan Request & Other Expenses")
        
        col4, col5, col6 = st.columns(3)

        with col4:
            requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000.0, value=1000000.0, step=10000.0)
            requested_emi = st.number_input("Target/Requested Monthly EMI (₹)", min_value=100.0, value=18000.0, step=100.0)
            emi_scenario = st.selectbox("EMI Calculation Scenario", ('Scenario A', 'Scenario B', 'Scenario C'))
            
        with col5:
            bank_balance = st.number_input("Bank Balance (₹)", min_value=0.0, value=500000.0, step=1000.0)
            emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0.0, value=150000.0, step=1000.0)
        
        with col6:
            school_fees = st.number_input("School Fees (₹)", min_value=0.0, value=5000.0, step=100.0)
            groceries_utilities = st.number_input("Groceries/Utilities (₹)", min_value=0.0, value=12000.0, step=100.0)
            other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0.0, value=5000.0, step=100.0)
            travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0.0, value=3000.0, step=100.0)
            college_fees = st.number_input("College Fees (₹)", min_value=0.0, value=0.0, step=100.0)

        st.markdown("---")
        submit_button = st.form_submit_button("Predict Loan Decision")


    if submit_button:
        # Map input to CustomData object
        data = CustomData(
            age=age, gender=gender, marital_status=marital_status, family_size=family_size, dependents=dependents,
            education=education, employment_type=employment_type, company_type=company_type, years_of_employment=years_of_employment,
            monthly_salary=monthly_salary, bank_balance=bank_balance, emergency_fund=emergency_fund, requested_amount=requested_amount,
            requested_tenure=requested_tenure, credit_score=credit_score, existing_loans=existing_loans, current_emi_amount=current_emi_amount,
            house_type=house_type, monthly_rent=monthly_rent, school_fees=school_fees, college_fees=college_fees, travel_expenses=travel_expenses,
            groceries_utilities=groceries_utilities, other_monthly_expenses=other_monthly_expenses, emi_scenario=emi_scenario
        )
        
        df_input = data.get_data_as_dataframe()
        
        # Run prediction
        result = pipeline.predict_loan_decision(df_input, requested_emi)
        
        st.markdown("---")
        st.header("Prediction Results")
        
        if result['Loan Eligibility'] == "ELIGIBLE":
            st.success(f"✅ Loan Eligibility: {result['Loan Eligibility']}")
        else:
            st.error(f"❌ Loan Eligibility: {result['Loan Eligibility']}")

        
        st.markdown(f"**Predicted Maximum Monthly EMI:** ₹{result['Predicted Max Monthly EMI']:,.2f}")
        st.markdown(f"**Requested Monthly EMI:** ₹{result['Requested EMI']:,.2f}")
        st.info(f"**Decision Reason:** {result['Decision Reason']}")


if __name__ == "__main__":
    main()