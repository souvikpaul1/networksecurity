import os
import sys
import json

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the MongoDB URL from environment variables
uri = os.getenv("MONGO_DB_URL")

# Create a new client and connect to the server
from pymongo.mongo_client import MongoClient
client = MongoClient(uri)

import certifi
ca= certifi.where() #secure http connection

import pymongo
import pandas as pd
import numpy as np
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())

            return records
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)    

    def insert_data_to_mongodb(self, records,database, collection_name):
        try:
            self.database = database
            self.collection_name = collection_name
            self.records = records
            # Ensure the MongoDB client is connected
            self.mongo_client=pymongo.MongoClient(uri, tlsCAFile=ca)

            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection_name]
            
            self.collection.insert_many(self.records)
            return(len(self.records))
       
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__ == "__main__":
    FILE_PATH="Network_data\data1.csv"
    DATABASE="network_security"
    COLLECTION_NAME="network_data"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_converter(FILE_PATH)
    print(records[:5])  # Print first 5 records for verification
    logging.info(f"Converting CSV data from {FILE_PATH} to JSON format.")
    inserted_count = networkobj.insert_data_to_mongodb(records, DATABASE, COLLECTION_NAME)
    print(f"Inserted {inserted_count} records into the MongoDB collection '{COLLECTION_NAME}' in database '{DATABASE}'.")