import sys
from typing import Tuple
import os
import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils import load_numpy_array_data, load_object, save_object,evaluate_models
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity  import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

from networksecurity.utils.ml_utils.model.estimator import MyModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier)
 

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        

    # def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
    #     """
    #     Method Name :   get_model_object_and_report
    #     Description :   This function trains a RandomForestClassifier with specified parameters
        
    #     Output      :   Returns metric artifact object and trained model object
    #     On Failure  :   Write an exception log and then raise an exception
    #     """
    #     try:
    #         logging.info("Training RandomForestClassifier with specified parameters")

    #         # Splitting the train and test data into features and target variables
    #         x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
    #         logging.info("train-test split done.")

    #         # Initialize RandomForestClassifier with specified parameters
    #         model = RandomForestClassifier(
    #             n_estimators = self.model_trainer_config._n_estimators,
    #             min_samples_split = self.model_trainer_config._min_samples_split,
    #             min_samples_leaf = self.model_trainer_config._min_samples_leaf,
    #             max_depth = self.model_trainer_config._max_depth,
    #             criterion = self.model_trainer_config._criterion,
    #             random_state = self.model_trainer_config._random_state
    #         )

    #         # Fit the model
    #         logging.info("Model training going on...")
    #         model.fit(x_train, y_train)
    #         logging.info("Model training done.")

    #         # Predictions and evaluation metrics
    #         y_pred = model.predict(x_test)
    #         accuracy = accuracy_score(y_test, y_pred)
    #         f1 = f1_score(y_test, y_pred)
    #         precision = precision_score(y_test, y_pred)
    #         recall = recall_score(y_test, y_pred)

    #         # Creating metric artifact
    #         metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
    #         return model, metric_artifact
        
    #     except Exception as e:
    #         raise NetworkSecurityException(e, sys) from e

    def track_in_mlflow(self, model: object, classification_train_metric: ClassificationMetricArtifact, classification_test_metric: ClassificationMetricArtifact) -> None:
        """
        Method Name :   track_in_mlflow
        Description :   This function tracks the model and metrics in MLflow
        
        Output      :   None
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_experiment("NetworkSecurity_Experiment")
            with mlflow.start_run():
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_metric("train_f1_score", classification_train_metric.f1_score)
                mlflow.log_metric("train_precision_score", classification_train_metric.precision_score)
                mlflow.log_metric("train_recall_score", classification_train_metric.recall_score)
                mlflow.log_metric("test_f1_score", classification_test_metric.f1_score)
                mlflow.log_metric("test_precision_score", classification_test_metric.precision_score)
                mlflow.log_metric("test_recall_score", classification_test_metric.recall_score)
                mlflow.sklearn.log_model(model, "trained_model")

            logging.info("Model and metrics tracked in MLflow successfully.")
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def train_model(self, x_train, y_train, x_test, y_test) -> object:
        models = {
            "LogisticRegression": LogisticRegression(verbose=1),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(verbose=1),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1)
        }


        params = {
            "DecisionTreeClassifier": {
                "criterion": ["gini", "entropy","log_loss"],
                # "max_depth": [None, 5, 10, 15],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4]
            },
            "RandomForestClassifier": {
                "n_estimators": [50, 100, 200],
                # "criterion": ["gini", "entropy","log_loss"],
                # "max_depth": [None, 5, 10, 15],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4]
            },
            "AdaBoostClassifier": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0]
            },  
            "GradientBoostingClassifier": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0],
                # "max_depth": [3, 5, 7],
                # "loss": ["log_loss", "exponential"]
                "subsample": [0.8, 0.9, 1.0]
            },  
            "LogisticRegression": {
                "penalty": ["l2", "none"],
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "sag", "saga"]
            },
            "KNeighborsClassifier": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree"]
            }   
        }

        model_report: dict = evaluate_models(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            models=models,
            params=params
        )


        best_model_score = max(sorted(model_report.values()))
        best_model_name=list(model_report.keys())[
                        list(model_report.values()).index(best_model_score)
                        
                    ]
        best_model= models[best_model_name]
        logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

        y_train_pred = best_model.predict(x_train)
        classification_train_metric=get_classification_score(y_true=y_train, y_pred=y_train_pred)
        logging.info(f"Model {best_model_name} trained successfully with training data.")
        y_test_pred = best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test, y_pred=y_test_pred)
        logging.info(f"Model {best_model_name} trained successfully with testing data.")
        logging.info(f"Model {best_model_name} trained successfully with training and testing data.")

        # function to track experiments in ML flow
        self.track_in_mlflow(best_model,classification_train_metric, classification_test_metric)

        # Load preprocessing object
        preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        logging.info("Preprocessing obj loaded.")

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True
                    )
        
        MyModel = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=MyModel)
        logging.info(f"Model {best_model_name} saved successfully at {self.model_trainer_config.trained_model_file_path}")
        logging.info(f"Model {best_model_name} trained and saved successfully.")

        save_object("final_models/model.pkl", best_model)
        


        #model trainer artifact:
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            classification_train_metric=classification_train_metric,
            classification_test_metric=classification_test_metric
        )
        logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
        return model_trainer_artifact
    





    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
            logging.info(f"Train target shape: {y_train.shape}, Test target shape: {y_test.shape}")

            model_trainer_artifact = self.train_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )
            return model_trainer_artifact
            
            # # Train model and get metrics
            # trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            # logging.info("Model object and artifact loaded.")
            
            # # Load preprocessing object
            # preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            # logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            # logging.info("Saving new model as performace is better than previous one.")
            # my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            # save_object(self.model_trainer_config.trained_model_file_path, my_model)
            # logging.info("Saved final model object that includes both preprocessing and the trained model")

            # # Create and return the ModelTrainerArtifact
            # model_trainer_artifact = ModelTrainerArtifact(
            #     trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            #     metric_artifact=metric_artifact,
            # )
            # logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            # return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e