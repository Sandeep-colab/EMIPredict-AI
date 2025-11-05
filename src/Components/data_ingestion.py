import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Define the target column for stratification
TARGET_COLUMN = 'emi_eligibility' 

# --- Configuration ---
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# --- Pre-Validation Cleaning Function (ULTRA-ROBUST TARGET FIX) ---
def clean_data_for_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Applies immediate cleaning with aggressive target fixation."""
    print("Starting Pre-Validation Data Cleaning.")
    
    # 1. Clean financial columns (unchanged)
    financial_cols = ['monthly_salary', 'bank_balance', 'emergency_fund', 'requested_amount', 'current_emi_amount', 
                      'monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                      'groceries_utilities', 'other_monthly_expenses', 'max_monthly_emi']
    
    for col in financial_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce') 
            
    # 2. Handle Age, Count, and Credit Score columns (unchanged, ensuring validation passes)
    if 'credit_score' in df.columns:
        df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce').fillna(650)
        df['credit_score'] = df['credit_score'].clip(300, 900)

    count_cols = ['age', 'existing_loans', 'family_size', 'dependents', 'years_of_employment', 'requested_tenure'] 
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col == 'age':
                 df[col] = df[col].fillna(18).clip(18, 100)
            else:
                 df[col] = df[col].fillna(0) 
            df[col] = df[col].astype(np.int64)

    # 3. CRITICAL FIX: Explicitly force non-zero target values to 1
    if TARGET_COLUMN in df.columns:
        # Step A: Convert to string, clean whitespace, and look for known positive markers
        target_series = df[TARGET_COLUMN].astype(str).str.strip().str.upper()

        # Step B: Identify positive examples. Assuming '1' or '1.0' or 'YES'/'TRUE' are positive.
        # Everything else (including NaN, 0, '0.0', 'NO', etc.) should be 0.
        positive_mask = target_series.isin(['1', '1.0', 'YES', 'Y', 'TRUE'])
        
        # Step C: Apply transformation
        df[TARGET_COLUMN] = 0
        df.loc[positive_mask, TARGET_COLUMN] = 1

        # Final check (should always pass now)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.int64)
        
    return df


# --- Data Ingestion Main Class (unchanged logic) ---
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() 

    def initiate_data_ingestion(self):
        print("Entered the data ingestion method or component")
        try:
            # Assuming data/raw/emi_prediction_dataset.csv exists
            df = pd.read_csv('data/raw/emi_prediction_dataset.csv', low_memory=False, encoding='latin1')
            print("Read the dataset as dataframe")

            # --- CRITICAL NEW STEP: CLEAN DATA IMMEDIATELY ---
            df = clean_data_for_validation(df)

            artifacts_dir = os.path.dirname(self.ingestion_config.raw_data_path)
            os.makedirs(artifacts_dir, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            print("Cleaned raw data saved to artifacts")

            # Perform Train-Test Split with STRATIFICATION
            target_unique_classes = df[TARGET_COLUMN].nunique()
            
            if target_unique_classes > 1:
                 print(f"Target has {target_unique_classes} classes. Using stratified split.")
                 train_set, test_set = train_test_split(
                    df, 
                    test_size=0.2, 
                    random_state=42,
                    stratify=df[TARGET_COLUMN]
                )
            else:
                # If this warning still appears, the data itself is likely 100% 0s
                print("WARNING: Target is single-class AFTER cleaning. Proceeding with non-stratified split.")
                train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
                
            # Save the split sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            print("Ingestion and split completed. Train/Test files saved.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            print(f"Error occurred during data ingestion: {e}")
            raise


if __name__ == "__main__":
    obj = DataIngestion()
    try:
        obj.initiate_data_ingestion()
        print("Data Ingestion and Pre-Cleaning complete. Ready for Data Validation check.")
    except Exception as e:
        print("Data Ingestion FAILED.")