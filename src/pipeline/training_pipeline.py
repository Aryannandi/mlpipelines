import os, sys
from src.logger import logging
from src.exception import CustmeException
from src.components.data_ingestion import DataIngestion
from src.components.data_transfromer import DataTransfromation
from src.components.modrl_trainer import ModelTrainer
from dataclasses import dataclass

if __name__ =="__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.inititate_data_ingestion()
    data_transformtion= DataTransfromation()
    train_arr, test_arr, _ = data_transformtion.inititate_data_transformation(train_data_path, test_data_path)
    model_training = ModelTrainer()
    model_training.inititate_model_trainer(train_arr, test_arr)


    # mlpipelines\src\pipeline\training_pipeline