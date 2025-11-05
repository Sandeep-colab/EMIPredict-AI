import os
import sys
import numpy as np
import pickle 
from dataclasses import dataclass

# Scikit-learn models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- Utility Functions ---
def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        print(f"Error saving object to {file_path}: {e}"); raise

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        print(f"Error loading object from {file_path}: {e}"); raise


# --- Configuration ---
@dataclass
class ModelTrainerConfig:
    train_array_path: str = os.path.join('artifacts', "train_transformed.npy")
    test_array_path: str = os.path.join('artifacts', "test_transformed.npy")
    
    # We still define the classification path, but the model saved will be None/Placeholder
    classification_model_path: str = os.path.join("artifacts", "best_classification_model.pkl") 
    regression_model_path: str = os.path.join("artifacts", "best_regression_model.pkl")

# Define the index of the target columns in the transformed array (51 columns total)
FEATURE_COUNT = 49 
TARGET_CLASS_INDEX = 49
TARGET_REG_INDEX = 50

# --- Model Trainer Class ---
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self):
        print("Entered model trainer method.")
        
        try:
            # 1. Load Transformed Data
            train_array = np.load(self.config.train_array_path)
            test_array = np.load(self.config.test_array_path)
            print("Loaded transformed train and test arrays.")
            
            # 2. Separate Features (X) and Targets (Y)
            X_train, y_class_train, y_reg_train = (
                train_array[:, :FEATURE_COUNT], 
                train_array[:, TARGET_CLASS_INDEX].astype(int),
                train_array[:, TARGET_REG_INDEX]
            )
            
            X_test, y_class_test, y_reg_test = (
                test_array[:, :FEATURE_COUNT], 
                test_array[:, TARGET_CLASS_INDEX].astype(int),
                test_array[:, TARGET_REG_INDEX]
            )

            print(f"Train Features Shape: {X_train.shape}. Targets separated.")

            # --- DIAGNOSTIC CONFIRMATION ---
            unique_classes, class_counts = np.unique(y_class_train, return_counts=True)
            print(f"Classification Target (emi_eligibility) unique values: {unique_classes}")
            print(f"Classification Target (emi_eligibility) class counts: {class_counts}")
            
            if len(unique_classes) <= 1:
                print("WARNING: Classification skipped due to single-class target. The source data contains no positive examples for eligibility.")
                # Skip classification logic and set placeholder results
                best_clf_model = "Classification Skipped (Single Class Target)"
            
            # 3. Define Regression Model Candidates
            regression_models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
            }
            
            # --- 4. REGRESSION: Training and Evaluation ---
            
            best_reg_score = -np.inf
            best_reg_model = None
            print("\nStarting Regression Model Training...")
            
            for name, model in regression_models.items():
                model.fit(X_train, y_reg_train)
                y_pred_reg = model.predict(X_test)
                
                r2 = r2_score(y_reg_test, y_pred_reg)
                print(f"  {name} R2 Score: {r2:.4f}")

                if r2 > best_reg_score:
                    best_reg_score = r2
                    best_reg_model = model
            
            print(f"\nBest Regression Model: {type(best_reg_model).__name__} with R2 Score: {best_reg_score:.4f}")

            # --- 5. Save Best Models ---
            # Save a placeholder for the classification model
            save_object(self.config.classification_model_path, best_clf_model) 
            save_object(self.config.regression_model_path, best_reg_model)
            print("Best classification placeholder and regression model saved to artifacts.")
            
            return (
                self.config.classification_model_path,
                self.config.regression_model_path,
            )

        except Exception as e:
            print(f"Error during Model Training: {e}"); raise

# -------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(ModelTrainerConfig.train_array_path):
        print("Error: Transformed NumPy arrays not found. Please run Data Transformation first.")
    else:
        try:
            trainer = ModelTrainer()
            clf_path, reg_path = trainer.initiate_model_trainer()
            
            print("\nModel Training successfully completed! ✅")
            print(f"Best Classification Model (Placeholder) saved to: {clf_path}")
            print(f"Best Regression Model saved to: {reg_path}")
            
        except Exception as e:
            print(f"Model Training FAILED: {e}")