import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constants.training_pipeline import TRAINING_BUCKET_NAME
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
# from src.components.model_evaluation import ModelEvaluation
# from src.components.model_pusher import ModelPusher

from networksecurity.entity.config_entity import (TrainingPipelineConfig,
                                        DataIngestionConfig,
                                        DataValidationConfig,
                                          DataTransformationConfig,
                                          ModelTrainerConfig)
                                          
from networksecurity.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact)

from networksecurity.cloud.s3_syncer import s3Sync

class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = s3Sync()  




    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            Data_Ingestion= DataIngestion(Data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifact = Data_Ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:

            self.data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            self.data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:

            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(self.model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    # save artifact to s3 bucket:
    def sync_artifacts_to_s3(self) -> None:
        """
        This method of TrainPipeline class is responsible for syncing artifacts to S3
        """
        try:
            logging.info("Syncing artifacts to S3")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
            logging.info("Artifacts synced to S3 successfully")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # save model to s3 bucket:    
    def sync_model_to_s3(self, model_trainer_artifact: ModelTrainerArtifact) -> None:
        """
        This method of TrainPipeline class is responsible for syncing model to S3
        """
        try:
            logging.info("Syncing model to S3")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir,aws_bucket_url=aws_bucket_url)
            logging.info("Model synced to S3 successfully")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    # def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
    #                            model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
    #     """
    #     This method of TrainPipeline class is responsible for starting modle evaluation
    #     """
    #     try:
    #         model_evaluation = ModelEvaluation(model_eval_config=self.model_evaluation_config,
    #                                            data_ingestion_artifact=data_ingestion_artifact,
    #                                            model_trainer_artifact=model_trainer_artifact)
    #         model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
    #         return model_evaluation_artifact
    #     except Exception as e:
    #         raise MyException(e, sys)

    # def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
    #     """
    #     This method of TrainPipeline class is responsible for starting model pushing
    #     """
    #     try:
    #         model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
    #                                    model_pusher_config=self.model_pusher_config
    #                                    )
    #         model_pusher_artifact = model_pusher.initiate_model_pusher()
    #         return model_pusher_artifact
    #     except Exception as e:
    #         raise MyException(e, sys)

    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            self.sync_artifacts_to_s3()
            self.sync_model_to_s3()

            return model_trainer_artifact
            # model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
            #                                                         model_trainer_artifact=model_trainer_artifact)
            # if not model_evaluation_artifact.is_model_accepted:
            #     logging.info(f"Model not accepted.")
            #     return None
            # model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)