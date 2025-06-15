from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact



if __name__ == "__main__":
    try:
        Training_Pipeline_Config = TrainingPipelineConfig()
        logging.info("Training Pipeline Config initialized successfully.")
        Data_Ingestion_Config = DataIngestionConfig(Training_Pipeline_Config)
        Data_Ingestion= DataIngestion(Data_Ingestion_Config)
        logging.info("Data Ingestion component initialized successfully.")
        Data_Ingestion_Artifact=Data_Ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed.")
        print(DataIngestionArtifact)
        data_validation_config=DataValidationConfig(Training_Pipeline_Config)
        data_validation = DataValidation(Data_Ingestion_Artifact,data_validation_config)
        logging.info("Data Validation component initialized successfully.")
        Data_Validation_Artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed.")
        print(Data_Validation_Artifact)


        data_transformation_config=DataTransformationConfig(Training_Pipeline_Config)
        data_transformation = DataTransformation(Data_Ingestion_Artifact,data_transformation_config)
        logging.info("Data Transformation component initialized successfully.")
        Data_transformation_Artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed.")
        print(Data_transformation_Artifact)


        logging.info("Model training started")
        model_trainer_config = ModelTrainerConfig(Training_Pipeline_Config)
        model_trainer = ModelTrainer(model_trainer_config, Data_transformation_Artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging

    except Exception as e:
        raise NetworkSecurityException(e, sys)