import os
import sys
import numpy as np
import pickle 
from dataclasses import dataclass
import mlflow             # <--- ADDED
import mlflow.sklearn     # <--- ADDED

# Scikit-learn models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- Utility Functions (Keep as is) ---
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
    
    # Classification model path kept for placeholder consistency
    classification_model_path: str = os.path.join("artifacts", "best_classification_model.pkl") 
    # Removed regression_model_path as MLflow manages it now.

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
            # 1. Load Transformed Data (Skipped section for brevity)
            train_array = np.load(self.config.train_array_path)
            test_array = np.load(self.config.test_array_path)
            
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

            # --- MLFLOW SETUP ---
            mlflow.set_experiment("EMI_Max_Affordability_Prediction")
            # --------------------

            # Start the MLflow Run
            with mlflow.start_run() as run: # <--- Start the run here

                # Diagnostics (Skipped section for brevity)
                unique_classes, class_counts = np.unique(y_class_train, return_counts=True)
                if len(unique_classes) <= 1:
                    print("WARNING: Classification skipped due to single-class target.")
                    best_clf_model = "Classification Skipped (Single Class Target)"
                
                # 3. Define Regression Model Candidates
                regression_models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), # Added params for logging
                }
                
                # --- 4. REGRESSION: Training and Evaluation ---
                best_reg_score = -np.inf
                best_reg_model = None
                best_model_name = ""
                
                print("\nStarting Regression Model Training...")
                
                for name, model in regression_models.items():
                    model.fit(X_train, y_reg_train)
                    y_pred_reg = model.predict(X_test)
                    
                    r2 = r2_score(y_reg_test, y_pred_reg)
                    print(f"  {name} R2 Score: {r2:.4f}")

                    if r2 > best_reg_score:
                        best_reg_score = r2
                        best_reg_model = model
                        best_model_name = name
                
                print(f"\nBest Regression Model: {best_model_name} with R2 Score: {best_reg_score:.4f}")

                # --- 5. Log and Save Best Model with MLflow ---
                
                # Log the final performance metric
                mlflow.log_metric("test_r2_score", best_reg_score)
                
                # Log the best model's type and any relevant parameters
                mlflow.set_tag("best_model_type", best_model_name)
                if best_model_name == "Random Forest Regressor":
                    mlflow.log_param("rf_n_estimators", 100)
                    mlflow.log_param("rf_max_depth", 10)
                
                # Log the model artifact itself
                mlflow.sklearn.log_model(
                    sk_model=best_reg_model, 
                    artifact_path="model" # This creates the URI: runs:/[RUN_ID]/model
                )
                
                # --- 6. Save Classification Placeholder (still uses pickle) ---
                save_object(self.config.classification_model_path, best_clf_model) 
                
                # We return the RUN ID instead of the regression model path
                return self.config.classification_model_path, run.info.run_id # <--- CHANGED RETURN
                
        except Exception as e:
            print(f"Error during Model Training: {e}"); raise

# -------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(ModelTrainerConfig.train_array_path):
        print("Error: Transformed NumPy arrays not found. Please run Data Transformation first.")
    else:
        try:
            trainer = ModelTrainer()
            clf_path, reg_run_id = trainer.initiate_model_trainer() # <--- CHANGED RECEIVING VARIABLE
            
            print("\nModel Training successfully completed! ✅")
            print(f"Best Classification Model (Placeholder) saved to: {clf_path}")
            print(f"Best Regression Model saved via MLflow. RUN ID: {reg_run_id}")
            
        except Exception as e:
            print(f"Model Training FAILED: {e}")