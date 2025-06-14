from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig
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

    except Exception as e:
        raise NetworkSecurityException(e, sys)