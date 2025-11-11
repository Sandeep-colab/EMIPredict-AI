import pandas as pd
import numpy as np
import os
import pickle
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
import mlflow 
import mlflow.pyfunc # <--- ADDED for loading MLflow models

# --- Configuration ---
# Get the directory of the current script (src/Components)
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Resolve the absolute path to the 'artifacts' directory at the project root
# Assuming the PredictionPipeline is run from the same project structure
_artifacts_dir = os.path.normpath(os.path.join(_script_dir, '..', '..', 'artifacts')) 

# --- CRITICAL CHANGE: MLflow URI for the Regression Model ---
# This MLflow URI must be dynamically set after the ModelTrainer has run.
# For demonstration, we will use a placeholder URI or an environment variable.
# In a real pipeline, the RUN_ID is passed from the trainer to the predictor.
# For now, we'll keep the CLF path and use a placeholder for the Regression Model.
# NOTE: In a real system, you would pass the RUN_ID at runtime or fetch the latest.

@dataclass
class PredictionPipelineConfig:
    preprocessor_path: str = os.path.join(_artifacts_dir, "preprocessor.pkl")
    classification_model_path: str = os.path.join(_artifacts_dir, "best_classification_model.pkl")
    # We will use an environment variable or default URI for the regression model
    # The actual URI (e.g., "runs:/...") must be provided at runtime.
    mlflow_tracking_uri: str = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns") # Set the tracking server
    
    # Placeholder for the Regression Model URI - MUST BE SET at runtime
    regression_model_uri: str = "MLFLOW_URI_NOT_SET" 


# --- Utility Function ---
def load_object(file_path):
    """Loads a Python object (like a model or preprocessor) from a binary file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Exception(f"Error loading object from {file_path}: {e}")

# --- Feature Engineering (Must match transformation) ---
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (Keep this function exactly as is from your previous code) ...
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


# --- Main Prediction Class ---
class CustomData:
    # ... (Keep this class exactly as is) ...
    """Helper class to structure applicant input data."""
    def __init__(self, **kwargs):
        self.data = kwargs
        
    def get_data_as_dataframe(self):
        df = pd.DataFrame([self.data])
        for col in ['monthly_salary', 'bank_balance', 'emergency_fund', 'requested_amount', 'current_emi_amount', 'credit_score', 'age', 'years_of_employment', 'family_size', 'dependents', 'requested_tenure']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        return df


class PredictionPipeline:
    
    # ACCEPT THE RUN_ID or full URI at initialization
    def __init__(self, regression_run_id_or_uri: str = None):
        self.config = PredictionPipelineConfig()
        
        # Determine the final regression model URI
        if regression_run_id_or_uri and not regression_run_id_or_uri.startswith("runs:/"):
             # Assume it's a RUN_ID and construct the URI
            self.config.regression_model_uri = f"runs:/{regression_run_id_or_uri}/model"
        elif regression_run_id_or_uri:
            # Assume it's the full URI
            self.config.regression_model_uri = regression_run_id_or_uri
        
        # Set MLflow tracking URI (important for loading remote models)
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)


    def load_artifacts(self):
        """
        Loads the preprocessor and both trained models/placeholders.
        Regression model is loaded using MLflow.
        """
        try:
            print("Attempting to load artifacts...")
            
            # Load Preprocessor (Still uses Pickle)
            preprocessor = load_object(self.config.preprocessor_path)
            
            # Load Classification Placeholder (Still uses Pickle)
            clf_model = load_object(self.config.classification_model_path)
            
            # Load Regression Model (NEW: Uses MLflow)
            if self.config.regression_model_uri == "MLFLOW_URI_NOT_SET":
                raise Exception("Regression Model MLflow URI was not provided.")
            
            print(f"Loading Regression Model from MLflow URI: {self.config.regression_model_uri}")
            reg_model = mlflow.pyfunc.load_model(self.config.regression_model_uri)
            
            return preprocessor, clf_model, reg_model
        except Exception as e:
            # Catch the error from load_object/mlflow and provide a context message
            raise Exception(f"Failed to load one or more artifacts: {e}")

    def predict_loan_decision(self, applicant_data: pd.DataFrame, requested_emi_amount: float) -> dict:
        # ... (Keep this function logic mostly as is, using the loaded artifacts) ...
        """
        Runs the dual-stage prediction logic:
        1. Predicts Max Monthly EMI.
        2. Compares Max EMI to Requested EMI to determine final eligibility.
        """
        
        # Load the necessary components
        preprocessor, clf_model_placeholder, reg_model = self.load_artifacts()

        # 1. Feature Engineering
        processed_df = add_engineered_features(applicant_data)
        
        # 2. Transformation
        data_scaled = preprocessor.transform(processed_df)

        # 3. Predict Max Monthly EMI (Regression Target)
        # NOTE: MLflow pyfunc models use model.predict(pandas_dataframe) by default.
        # However, since your model was logged as sklearn, it should accept the NumPy array.
        max_monthly_emi = reg_model.predict(data_scaled)[0] 
        max_monthly_emi = max_monthly_emi if max_monthly_emi > 0 else 0 

        # 4. Determine Final Decision based on Regression Output
        final_decision = "Ineligible (Max EMI < Requested EMI)"
        loan_eligible = 0
        
        if max_monthly_emi >= requested_emi_amount:
            final_decision = "Eligible (Max EMI >= Requested EMI)"
            loan_eligible = 1
        
        return {
            "Applicant Status": "Single-Class Classification Warning: Model uses Regression Fallback.",
            "Loan Eligibility": "ELIGIBLE" if loan_eligible == 1 else "INELIGIBLE",
            "Predicted Max Monthly EMI": round(max_monthly_emi, 2),
            "Requested EMI": requested_emi_amount,
            "Decision Reason": final_decision
        }


if __name__ == "__main__":
    
    # --- IMPORTANT: Replace with the actual RUN ID obtained from ModelTrainer ---
    # The RUN ID from your trainer's output needs to be placed here!
    # Example: reg_run_id = "18a988f2a822498098b2545646b54ffa" 
    
    # You need to run the ModelTrainer first and copy its output RUN ID here:
    print("\n--- WARNING: RUN ID must be manually provided after training! ---")
    reg_run_id = input("Please paste the Regression Model RUN ID here (e.g., 18a988f2a822...): ")
    # -------------------------------------------------------------------------
    
    try:
        # Initialize the pipeline with the MLflow RUN ID
        pipeline = PredictionPipeline(regression_run_id_or_uri=reg_run_id)

        # --- Sample Applicant Data ---
        applicant_data = CustomData(
            age=35, gender='Male', marital_status='Married', family_size=4, dependents=2,
            education='Graduate', employment_type='Salaried', company_type='Service',
            years_of_employment=8, monthly_salary=115000.0, bank_balance=500000.0,
            emergency_fund=150000.0, requested_amount=1000000.0, requested_tenure=60,
            credit_score=780.0, existing_loans=1, current_emi_amount=15000.0,
            house_type='Rented', monthly_rent=10000.0, school_fees=5000.0,
            college_fees=0.0, travel_expenses=3000.0, groceries_utilities=12000.0,
            other_monthly_expenses=5000.0, emi_scenario='Scenario A'
        )
    
        requested_emi = 18000.0
        
        # Run Prediction
        applicant_df = applicant_data.get_data_as_dataframe()
        final_decision = pipeline.predict_loan_decision(applicant_df, requested_emi)
        
        print("\n--- Loan Decision Prediction Output ---")
        print(pd.Series(final_decision).to_markdown())

    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"The prediction pipeline failed due to a critical error: {e}")
        print("Ensure the 'artifacts' files and the correct MLflow RUN ID were provided.")