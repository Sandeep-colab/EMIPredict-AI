import sys
import os
import numpy as np
import pandas as pd
import pickle 
from dataclasses import dataclass

# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# --- Utility Function (Mocking src/utils.py) ---
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        print(f"Error saving object to {file_path}: {e}"); raise


# --- Configuration ---
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    train_array_path: str = os.path.join('artifacts', "train_transformed.npy") 
    test_array_path: str = os.path.join('artifacts', "test_transformed.npy")
    TARGET_CLASSIFICATION = 'emi_eligibility'
    TARGET_REGRESSION = 'max_monthly_emi'


# --- Custom Feature Engineering Functions ---
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
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


# --- Preprocessing Pipeline Builder ---
def get_data_transformer_object():
    
    try:
        numerical_features = [
            'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount', 
            'requested_tenure', 'existing_loans', 'DTI_ratio', 'ETI_ratio', 
            'free_cash_flow', 'income_per_capita', 'income_per_dependent', 
            'job_stability_ratio', 'monthly_salary', 'age', 'years_of_employment', 
            'family_size', 'dependents', 'current_emi_amount', 'total_monthly_expenses'
        ]
        
        nominal_features = [
            'gender', 'marital_status', 'education', 'employment_type', 
            'company_type', 'house_type', 'emi_scenario', 
        ]
        
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ])

        nominal_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
        ])

        preprocessor = ColumnTransformer(
            [
                ("Numeric_Pipeline", numeric_pipeline, numerical_features),
                ("Nominal_Pipeline", nominal_pipeline, nominal_features),
            ],
            remainder='drop' 
        )

        return preprocessor

    except Exception as e:
        print(f"Error in data transformer object: {e}"); raise

# --- Data Transformation Main Class ---
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        print("Entered data transformation method")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            train_df = add_engineered_features(train_df)
            test_df = add_engineered_features(test_df)
            
            TARGET_CLASS = self.data_transformation_config.TARGET_CLASSIFICATION
            TARGET_REG = self.data_transformation_config.TARGET_REGRESSION
            
            input_feature_train_df = train_df.drop(columns=[TARGET_CLASS, TARGET_REG], axis=1)
            target_classification_train_df = train_df[TARGET_CLASS]
            target_regression_train_df = train_df[TARGET_REG]

            input_feature_test_df = test_df.drop(columns=[TARGET_CLASS, TARGET_REG], axis=1)
            target_classification_test_df = test_df[TARGET_CLASS]
            target_regression_test_df = test_df[TARGET_REG]

            preprocessing_obj = get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_classification_train_df), 
                np.array(target_regression_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_classification_test_df), 
                np.array(target_regression_test_df)
            ]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)
            np.save(self.data_transformation_config.train_array_path, train_arr)
            np.save(self.data_transformation_config.test_array_path, test_arr)
            print("Preprocessor object and transformed train/test arrays successfully saved to artifacts.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            print(f"Error in data transformation process: {e}"); raise


if __name__ == "__main__":
    TRAIN_FILE_PATH = os.path.join('artifacts', "train.csv")
    TEST_FILE_PATH = os.path.join('artifacts', "test.csv")
    
    if os.path.exists(TRAIN_FILE_PATH) and os.path.exists(TEST_FILE_PATH):
        try:
            data_transformer = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformer.initiate_data_transformation(
                train_path=TRAIN_FILE_PATH,
                test_path=TEST_FILE_PATH
            )
            print("\nData Transformation successfully completed! 🎉")
            print(f"Transformed Training Array Shape: {train_arr.shape}")
            print(f"Transformed Testing Array Shape: {test_arr.shape}")
            print(f"Preprocessor saved to: {preprocessor_path}")
        except Exception as e:
            print(f"\n--- CRITICAL ERROR DURING DATA TRANSFORMATION ---")
            print(f"Error: {e}")