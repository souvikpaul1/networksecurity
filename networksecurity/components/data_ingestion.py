from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import pandas as pd
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

# Fetch the MongoDB URL from environment variables
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            # training_pipeline_config = ...  # create or fetch this object
            # data_ingestion_config = DataIngestionConfig(training_pipeline_config)
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Method Name :   export_collection_as_dataframe
        Description :   This method exports the data from the specified MongoDB collection into a pandas DataFrame
        
        Output      :   DataFrame containing the data from the specified collection
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered export_collection_as_dataframe method of Data_Ingestion class")

        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name  
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)  
            collection=self.mongo_client[database_name][collection_name]

            print("DB:", database_name, "Collection:", collection_name)
            print("MongoDB URL:", MONGO_DB_URL)

            df=pd.DataFrame(list(collection.find())) 
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"], axis=1)
            df.replace({"na": "np.nan"}, inplace=True)

            if df.empty:
                logging.warning("The DataFrame created from MongoDB is empty.")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from MongoDB to a CSV file in the feature store directory
        
        Output      :   DataFrame containing the data exported to the feature store
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered export_data_into_feature_store method of Data_Ingestion class")

        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
        
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            Data_Ingestion_Artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info("Got the data from mongodb")

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            
            logging.info(f"Data ingestion artifact: {Data_Ingestion_Artifact}")
            return Data_Ingestion_Artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        