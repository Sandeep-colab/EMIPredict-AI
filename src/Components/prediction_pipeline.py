import pandas as pd
import numpy as np
import os
import pickle
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
@dataclass
class PredictionPipelineConfig:
    preprocessor_path: str = os.path.join('artifacts', "preprocessor.pkl")
    classification_model_path: str = os.path.join("artifacts", "best_classification_model.pkl")
    regression_model_path: str = os.path.join("artifacts", "best_regression_model.pkl")

# --- Utility Function ---
def load_object(file_path):
    """Loads a Python object (like a model or preprocessor) from a binary file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        print(f"Error loading object from {file_path}: {e}")
        raise

# --- Feature Engineering (Must match transformation) ---
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all the custom feature engineering logic."""
    df_copy = df.copy()
    
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                    'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount']
    
    # Fill with 0 for calculation robustness
    for col in expense_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
    
    df_copy['total_monthly_expenses'] = df_copy[expense_cols].sum(axis=1)
    
    # Handle division by zero for ratio stability
    safe_salary = pd.to_numeric(df_copy['monthly_salary'], errors='coerce').replace(0, 1)

    df_copy['DTI_ratio'] = df_copy['current_emi_amount'] / safe_salary
    df_copy['ETI_ratio'] = df_copy['total_monthly_expenses'] / safe_salary
    df_copy['free_cash_flow'] = df_copy['monthly_salary'] - df_copy['total_monthly_expenses']

    df_copy['income_per_capita'] = df_copy['monthly_salary'] / df_copy['family_size'].replace(0, 1)
    df_copy['income_per_dependent'] = df_copy['monthly_salary'] / (df_copy['dependents'].replace(0, 0) + 1)
    
    df_copy['job_stability_ratio'] = df_copy['years_of_employment'] / df_copy['age'].replace(0, 1)
    
    return df_copy


# --- Main Prediction Class ---
class CustomData:
    """Helper class to structure applicant input data."""
    def __init__(self, **kwargs):
        self.data = kwargs
        
    def get_data_as_dataframe(self):
        df = pd.DataFrame([self.data])
        # Force numeric types for pre-cleaning, similar to ingestion
        for col in ['monthly_salary', 'bank_balance', 'emergency_fund', 'requested_amount', 'current_emi_amount', 'credit_score', 'age', 'years_of_employment', 'family_size', 'dependents', 'requested_tenure']:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()
        
    def load_artifacts(self):
        """Loads the preprocessor and both trained models/placeholders."""
        try:
            preprocessor = load_object(self.config.preprocessor_path)
            clf_model = load_object(self.config.classification_model_path)
            reg_model = load_object(self.config.regression_model_path)
            return preprocessor, clf_model, reg_model
        except Exception as e:
            print(f"Failed to load one or more artifacts: {e}")
            raise

    def predict_loan_decision(self, applicant_data: pd.DataFrame, requested_emi_amount: float) -> dict:
        """
        Runs the dual-stage prediction logic:
        1. Predicts Max Monthly EMI.
        2. Compares Max EMI to Requested EMI to determine final eligibility.
        
        Note: The classification step is commented out as it is currently non-functional due to source data issues.
        """
        
        # Load the necessary components
        preprocessor, clf_model_placeholder, reg_model = self.load_artifacts()

        # 1. Feature Engineering
        # We must ensure the input applicant data has the 49 features used for training
        processed_df = add_engineered_features(applicant_data)
        
        # 2. Transformation
        # Use the fitted preprocessor to transform the new data
        data_scaled = preprocessor.transform(processed_df)

        # 3. Predict Max Monthly EMI (Regression Target)
        max_monthly_emi = reg_model.predict(data_scaled)[0]
        max_monthly_emi = max_monthly_emi if max_monthly_emi > 0 else 0 # Ensure non-negative

        # 4. Determine Final Decision based on Regression Output
        final_decision = "Ineligible (Max EMI < Requested EMI)"
        loan_eligible = 0
        
        # The classification step is logically skipped due to single-class source data.
        # if clf_model_placeholder == "Classification Skipped (Single Class Target)":
        #    # Fallback to pure regression logic when classification fails
        if max_monthly_emi >= requested_emi_amount:
            final_decision = "Eligible"
            loan_eligible = 1
        
        return {
            "Applicant Status": "Single-Class Classification Warning: Model uses Regression Fallback.",
            "Loan Eligibility": "ELIGIBLE" if loan_eligible == 1 else "INELIGIBLE",
            "Predicted Max Monthly EMI": round(max_monthly_emi, 2),
            "Requested EMI": requested_emi_amount,
            "Decision Reason": final_decision
        }


if __name__ == "__main__":
    # --- Example Usage ---
    pipeline = PredictionPipeline()

    # --- Sample Applicant Data (Make sure to include all required raw columns) ---
    applicant_data = CustomData(
    age=35,
    gender='Male',
    marital_status='Married',
    family_size=4,
    dependents=2,
    education='Graduate',
    employment_type='Salaried',
    company_type='Service',
    years_of_employment=8,
    # <<< MODIFIED KEY INPUT >>>
    monthly_salary=115000.0, 
    # <<< MODIFIED KEY INPUT END >>>
    bank_balance=500000.0,
    emergency_fund=150000.0,
    requested_amount=1000000.0,
    requested_tenure=60,
    credit_score=780.0,
    existing_loans=1,
    current_emi_amount=15000.0,
    house_type='Rented',
    monthly_rent=10000.0,
    school_fees=5000.0,
    college_fees=0.0,
    travel_expenses=3000.0,
    groceries_utilities=12000.0,
    other_monthly_expenses=5000.0,
    emi_scenario='Scenario A',
    # NOTE: 'emi_eligibility' and 'max_monthly_emi' are targets and should NOT be in the input
)
 
# Applicant requests an EMI of 18,000
    requested_emi = 18000.0
    
    # Run Prediction
    applicant_df = applicant_data.get_data_as_dataframe()
    final_decision = pipeline.predict_loan_decision(applicant_df, requested_emi)
    
    print("\n--- Loan Decision Prediction ---")
    print(pd.Series(final_decision).to_markdown())