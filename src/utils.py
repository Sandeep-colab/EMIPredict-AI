# src/utils.py

import os
import sys
import pickle

# Assuming CustomeException is defined elsewhere
# from src.exception import CustomeException

def save_object(file_path, obj):
    """Saves a Python object (like a model or preprocessor) to a binary file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # raise CustomeException(e, sys)
        print(f"Error saving object to {file_path}: {e}")
        raise