import os
import sys
from sklearn.model_selection import GridSearchCV
import numpy as np
import dill
import yaml
from pandas import DataFrame
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,r2_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# def evaluate_models(x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame, model,param) -> float:
#     """
#     Evaluate the model using the training and testing data.
#     Returns the accuracy score of the model.
#     """
#     try:
#        report = {}
#        for i in range(len(list(model))):
#             model=list(model.values())[i]
#             para=param[list(model.keys())[i]]

#             gs=GridSearchCV(model,para,cv=3,verbose=2,n_jobs=-1)
#             gs.fit(x_train,y_train)

#             y_train_pred= gs.predict(x_train)
#             y_test_pred = gs.predict(x_test)

#             train_model_score = r2_score(y_train, y_train_pred)
#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(model.keys())[i]] = test_model_score

#             return report
    
#     except Exception as e:
#         raise NetworkSecurityException(e, sys) from e  
# 
def evaluate_models(
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    x_test: np.ndarray, 
    y_test: np.ndarray,
    models: dict,
    params: dict
) -> dict:
    """
    Evaluate multiple models using GridSearchCV.
    
    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Testing features
        y_test: Testing labels
        models: Dictionary of models to evaluate
        params: Dictionary of parameters for each model
        
    Returns:
        Dictionary containing model evaluation results
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            # Get parameters for current model
            para = params[model_name]
            
            # Perform GridSearch
            gs = GridSearchCV(model, para, cv=3, verbose=2, n_jobs=-1)
            gs.fit(x_train, y_train)
            
            # Make predictions
            y_train_pred = gs.predict(x_train)
            y_test_pred = gs.predict(x_test)
            
            # Calculate scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store results
            report[model_name] = {
                'train_score': train_model_score,
                'test_score': test_model_score,
                'best_params': gs.best_params_,
                'best_score': gs.best_score_
            }
            
        return report
        
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e 
    