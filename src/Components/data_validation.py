import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any

# --- Configuration ---
@dataclass
class DataValidationConfig:
    """Configuration class for Data Validation paths and status."""
    report_file_path: str = os.path.join('artifacts', "data_validation_report.txt")
    validation_status: bool = False
    
    # --- SCHEMA DECLARATION ---
    SCHEMA: Dict[str, str] = field(default_factory=lambda: {
        # Core Numeric/Financial
        'credit_score': 'float64', 
        'monthly_salary': 'float64', 
        'age': 'int64',          
        'years_of_employment': 'int64', 
        'bank_balance': 'float64',
        'emergency_fund': 'float64',
        'requested_amount': 'float64',
        'requested_tenure': 'int64',
        'current_emi_amount': 'float64',
        
        # Target Variables
        'emi_eligibility': 'int64',      
        'max_monthly_emi': 'float64',    
        
        # Categorical columns
        'gender': 'object',
        'marital_status': 'object',
        'education': 'object',
        'employment_type': 'object',
        'company_type': 'object',
        'house_type': 'object',
        
        # Additional Columns
        'other_monthly_expenses': 'float64', 
        'college_fees': 'float64', 
        'groceries_utilities': 'float64',
        'travel_expenses': 'float64',
        'school_fees': 'float64',
        'monthly_rent': 'float64',
        'existing_loans': 'int64',       
        'emi_scenario': 'object', 
        
        # Other raw columns
        'family_size': 'int64',          
        'dependents': 'int64',           
    })

    # Define range constraints for critical numeric features 
    VALUE_CONSTRAINTS: Dict[str, tuple] = field(default_factory=lambda: {
        'credit_score': (300, 900),
        'age': (18, 100),
        'monthly_salary': (0, 1000000), 
        'requested_amount': (1000, 5000000)
    })

# --- Validation Class ---
class DataValidation:
    def __init__(self):
        self.config = DataValidationConfig()
        self.validation_status = True
        self.report_lines = ["--- Data Validation Report ---"]

    def check_schema(self, df: pd.DataFrame, dataset_name: str):
        """Checks for missing columns and correct data types."""
        
        expected_cols = set(self.config.SCHEMA.keys())
        actual_cols = set(df.columns)
        
        self.report_lines.append(f"\n[Dataset: {dataset_name}] Shape: {df.shape}")

        # Check Data Types
        for col, expected_dtype in self.config.SCHEMA.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if expected_dtype not in actual_dtype: 
                    self.report_lines.append(
                        f"❌ FAIL: {dataset_name} column '{col}' has type {actual_dtype}, expected {expected_dtype}"
                    )
                    self.validation_status = False

    def check_value_ranges(self, df: pd.DataFrame, dataset_name: str):
        """Checks if numeric data falls within acceptable business ranges."""
        
        for col, (min_val, max_val) in self.config.VALUE_CONSTRAINTS.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                outliers_low = df[df[col] < min_val].shape[0]
                outliers_high = df[df[col] > max_val].shape[0]
                total_outliers = outliers_low + outliers_high
                
                if total_outliers > 0:
                    self.report_lines.append(
                        f"❌ FAIL: {dataset_name} column '{col}' has {total_outliers} outliers (Range: {min_val}-{max_val})"
                    )
                    self.validation_status = False
                
    def initiate_data_validation(self, train_path: str, test_path: str) -> bool:
        print("Starting Data Validation process.")
        
        try:
            train_df = pd.read_csv(train_path, low_memory=False, encoding='latin1')
            test_df = pd.read_csv(test_path, low_memory=False, encoding='latin1')

            self.check_schema(train_df, "Train Set")
            self.check_value_ranges(train_df, "Train Set")
            
            self.check_schema(test_df, "Test Set")
            self.check_value_ranges(test_df, "Test Set")

            self.report_lines.append("\n--- Validation Summary ---")
            if self.validation_status:
                self.report_lines.append("✅ SUCCESS: All validation checks passed.")
            else:
                self.report_lines.append("❌ FAIL: One or more critical checks failed.")
                
            os.makedirs(os.path.dirname(self.config.report_file_path), exist_ok=True)
            with open(self.config.report_file_path, "w", encoding='utf-8') as f:
                f.write("\n".join(self.report_lines))

            print(f"Data Validation Complete. Status: {self.validation_status}")
            
            return self.validation_status

        except Exception as e:
            print(f"Critical error during validation process: {e}")
            self.validation_status = False
            return False

# -------------------------------------------------------------
if __name__ == "__main__":
    TRAIN_FILE_PATH = os.path.join('artifacts', "train.csv")
    TEST_FILE_PATH = os.path.join('artifacts', "test.csv")
    
    validator = DataValidation()
    
    validation_passed = validator.initiate_data_validation(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH
    )

    if validation_passed:
        print("\nPipeline Status: ✅ Data is Valid. Proceed to Transformation.")
    else:
        print("\nPipeline Status: ❌ Data Validation Failed. Check the report and fix data issues.")