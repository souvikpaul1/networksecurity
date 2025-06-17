from fastapi import FastAPI, Request, File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run
import pandas as pd

from typing import Optional

# Importing constants and pipeline modules from the project
# from src.constants import APP_HOST, APP_PORT
# from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from networksecurity.pipeline.training_pipeline import TrainPipeline
import sys
import os
import certifi

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils import  load_object

ca= certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print("MongoDB URL:", mongo_db_url)

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]




# Initialize FastAPI application
app = FastAPI()

# # Mount the 'static' directory for serving static files (like CSS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Set up Jinja2 template engine for rendering HTML templates
# templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Route to render the main page with the form



# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    """
    Renders the main HTML form page for vehicle data input.
    """
    return RedirectResponse(url="/docs")

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")



# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)