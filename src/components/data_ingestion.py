import pandas as pd
import numpy as np
import os,sys
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustmeException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # C:\Users\Aryan\Desktop\mlpipe\mlpipelines\notbook\data\Canada_per_capita_income (1).csv
    # mlpipelines\notbook\data
    # mlpipelines\notbook\data\Canada_per_capita_income (1).csv
    def inititate_data_ingestion(self):
        try:
            data = pd.read_csv(os.path.join("notbook/data", "Canada_per_capita_income (1).csv"))
            logging.info("Data Ingestion Started")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw Data Saved")

            train_set, test_set = train_test_split(data, test_size = 0.3, random_state=42)
            logging.info("Data Split Completed")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                # self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustmeException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.inititate_data_ingestion()


