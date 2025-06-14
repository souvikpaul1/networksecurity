from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys
from networksecurity.entity.artifact_entity import DataIngestionArtifact



if __name__ == "__main__":
    try:
        TrainingPipelineConfig = TrainingPipelineConfig()
        logging.info("Training Pipeline Config initialized successfully.")
        DataIngestionConfig = DataIngestionConfig(TrainingPipelineConfig)
        DataIngestion= DataIngestion(DataIngestionConfig)
        logging.info("Data Ingestion component initialized successfully.")
        DataIngestionArtifact=DataIngestion.initiate_data_ingestion()
        print(DataIngestionArtifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)